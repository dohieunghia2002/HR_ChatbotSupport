from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import unicodedata
import re

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===========================
# CONFIG
# ===========================
DATA_DIR = "data2/"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCqnz7-HFT06k6rmIgBVRRxV2-GzwQaR9I"

app = Flask(__name__)
CORS(app)

# ===========================
# TEXT CLEANING FUNCTIONS
# ===========================
def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("•", "- ").replace("▪", "- ").replace("·", "- ").replace("●", "- ").replace("✓", "- ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def clean_markdown(text):
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
            
        line = re.sub(r"^\*\s*\*\*(.+?)\*\*", r"- **\1**", line)
        line = re.sub(r"^\*\s+", "- ", line)

        if line.startswith("**") and not line.startswith("- **"):
            line = "- " + line
        
        cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Ensured directory exists: {path}")

# ===========================
# VECTORSTORE INIT
# ===========================
def create_or_load_vectorstore():
    ensure_directory(FAISS_INDEX_PATH)

    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print("Loading vector store...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded.")
    else:
        print("Building new vector store...")

        loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

        if len(documents) == 0:
            raise ValueError("No PDF files found in DATA_DIR.")

        print(f"Loaded {len(documents)} PDF files")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=520,
            chunk_overlap=60,
            separators=["\n\n", "\n", "✓", "-", "•", ".", " "]
        )
        docs = text_splitter.split_documents(documents)

        print(f"Split into {len(docs)} chunks")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)

        vectorstore.save_local(FAISS_INDEX_PATH)
        print("Vector store saved.")

    return vectorstore


# ===========================
# ENGLISH SYSTEM PROMPT
# ===========================
system_prompt = """
You are a professional HR Expert.
Your task is to answer questions ONLY based on the content of the provided CV.
If the information does not exist in the CV, respond: "This information does not appear in the CV."

Requirements:
- Always answer in **English only**.
- Keep responses short and clear.
- Use bullet points when appropriate.
- Focus on analyzing skills, experience, seniority level, relevance to job description, etc.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + """

==== CV Content ====
{context}
====================

Question: {question}
"""
)

# ===========================
# INIT MODEL & RETRIEVER
# ===========================
print("🔧 Initializing FAISS...")
vectorstore = create_or_load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5}
)

print("🔧 Initializing Gemini model...")
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=1024
)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

print("🔥 RAG chatbot is ready!")


# ===========================
# FLASK API
# ===========================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    result = qa_chain.invoke({"query": question})

    raw_answer = result["result"]
    answer = clean_markdown(raw_answer)

    sources = []
    for doc in result["source_documents"]:
        src = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", "N/A")
        sources.append({"file": src, "page": page})

    return jsonify({"answer": answer, "sources": sources}), 200


@app.route("/", methods=["GET"])
def home():
    return "RAG Chatbot Backend is running (English-only mode).", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
