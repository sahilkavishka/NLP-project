import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Initialize embeddings and load the existing Vector Database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_path = "./chroma_db_store"
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# 2. Configure the Retriever to fetch the top 2 most relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 3. Setup a local LLM using HuggingFace Pipeline (No API keys needed)
print("Loading local LLM... (This may take a minute on the first run to download the small model)")
hf_pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_length=150)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 4. Create the Prompt Template to instruct the AI
prompt_template = """
Use the provided context to answer the user's question accurately.
If the context does not contain the answer, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {input}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# 5. Build the RAG (Retrieval-Augmented Generation) Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 6. Test the System with a query based on the previously saved notes
query = "What are the 7 layers of the OSI model?"
print(f"\nQuery: {query}")
print("Searching knowledge base and generating answer...\n")

result = retrieval_chain.invoke({"input": query})

print("========================================")
print("Final Answer:")
print(result["answer"])
print("========================================")
