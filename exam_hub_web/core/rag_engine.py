import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma  # Updated to the new standalone package
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain  # Updated to langchain_classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # Updated to langchain_classic
from transformers import pipeline

class LectureRAG:
    """
    A professional module for Retrieval-Augmented Generation (RAG).
    Handles vector database creation and Q&A using local LLMs.
    """
    def __init__(self, db_dir="../data/chroma_db", embedding_model="all-MiniLM-L6-v2", llm_model="google/flan-t5-small"):
        print("[LectureRAG] Initializing Embeddings and LLM...")
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load Local LLM
        hf_pipe = pipeline("text2text-generation", model=llm_model, max_length=150)
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        
        self.vector_db = None
        print("[LectureRAG] Initialization complete.")

    def build_knowledge_base(self, text_content):
        """
        Takes raw transcript text, chunks it, and saves it to ChromaDB.
        """
        print(f"[LectureRAG] Building Knowledge Base at '{self.db_dir}'...")
        
        # Clear existing DB for a fresh start
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
            
        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        
        from langchain_core.documents import Document
        docs = [Document(page_content=text_content)]
        chunks = text_splitter.split_documents(docs)
        
        # Create and persist the database
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_dir
        )
        print(f"[LectureRAG] Successfully stored {len(chunks)} chunks in Vector DB.")

    def ask_question(self, question):
        """
        Queries the vector database and generates an answer using the LLM.
        """
        if not self.vector_db:
            if os.path.exists(self.db_dir):
                self.vector_db = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            else:
                return "Error: Knowledge base has not been built yet."
                
        print(f"[LectureRAG] Retrieving context for query: '{question}'...")
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})
        
        prompt_template = """Use the provided context to answer the user's question accurately.
Context:
{context}

Question: {input}

Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Build the LangChain execution pipeline
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Execute query
        result = retrieval_chain.invoke({"input": question})
        return result["answer"]

# ==========================================
# Module Testing Block
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    sample_transcript = "Welcome to Exam Hub. Today we will focus on Data Warehousing and Computer Networks for the 2026 syllabus."
    
    rag = LectureRAG()
    rag.build_knowledge_base(sample_transcript)
    
    test_q = "What subjects are we focusing on today?"
    answer = rag.ask_question(test_q)
    
    print("\n--- Q&A RESULT ---")
    print(f"Q: {test_q}")
    print(f"A: {answer}")
    print("------------------\n")