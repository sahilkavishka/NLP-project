import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. පරීක්ෂා කිරීම සඳහා මූලික සටහනක් (Text file) ස්වයංක්‍රීයව නිර්මාණය කිරීම
sample_text = """
Computer Networks: 
A computer network is a set of computers sharing resources located on or provided by network nodes.
The OSI model has 7 layers: Physical, Data Link, Network, Transport, Session, Presentation, Application.

Data Warehousing:
A data warehouse is a centralized repository of integrated data from one or more disparate sources.
It is used for reporting and data analysis, and is considered a core component of business intelligence.
"""
file_name = "exam_notes.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(sample_text)

print("1. Text folder create.")


loader = TextLoader(file_name)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)
print(f"2. text small parts (Chunks) {len(chunks)} devided.")

# 4. Embeddings Model එක Load කිරීම (HuggingFace හි ඇති වේගවත් Local Model එකක්)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. ChromaDB Vector Database එක නිර්මාණය කිරීම සහ දත්ත ගබඩා කිරීම
db_path = "./chroma_db_store"

# පරණ DB එකක් ඇත්නම් එය මකා දැමීම (පැහැදිලිව අලුතින් පටන් ගැනීමට)
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# Chunks ටික Vector DB එකට ඇතුළත් කර Save (Persist) කිරීම
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_path
)

print(f"3. ✅ data as  '{db_path}' store Vector Database !")
