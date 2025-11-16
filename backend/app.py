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
os.environ["GOOGLE_API_KEY"] = "AIzaSyBhadhUvpKhISbb4-91_IqdVm10dXZTjXo"

app = Flask(__name__)
CORS(app)

# ===========================
# Functions for VectorStore
# ===========================
def clean_text(text):
    # chuẩn hóa unicode
    text = unicodedata.normalize("NFC", text)

    # thay thế bullet
    text = text.replace("•", "- ").replace("▪", "- ").replace("·", "- ").replace("●", "- ").replace("✓", "- ")

    # xóa khoảng trắng thừa
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def clean_markdown(text):
    """
    Chuẩn hóa Markdown trong phản hồi:
    - Thay * **text** → - **text**
    - Thay * text → - text
    - Loại bỏ * thừa
    - Đảm bảo bullet dùng dấu gạch đầu dòng '-'
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Bỏ dòng trống
        if not line:
            cleaned_lines.append("")
            continue
            
        # Sửa: * **text** → - **text**
        line = re.sub(r"^\*\s*\*\*(.+?)\*\*", r"- **\1**", line)
        
        # Sửa: * text → - text
        line = re.sub(r"^\*\s+", "- ", line)
        
        # Sửa: **text** (không có bullet) → - **text**
        if line.startswith("**") and not line.startswith("- **"):
            line = "- " + line
        
        cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines)
    
    # Loại bỏ khoảng trắng thừa giữa các dòng
    result = re.sub(r"\n{3,}", "\n\n", result)
    
    return result.strip()

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Đã đảm bảo thư mục tồn tại: {path}")


def create_or_load_vectorstore():
    ensure_directory(FAISS_INDEX_PATH)

    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print("Đang tải lại vector store...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True
        )
        print("Tải thành công.")
    else:
        print("Không tìm thấy vector store. Đang tạo mới...")

        loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
        if len(documents) == 0:
            raise ValueError("Không tìm thấy file PDF nào trong DATA_DIR.")

        print(f"Đã nạp {len(documents)} tài liệu PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=30,
            separators=["\n\n", "\n", "✓", "-", "•", ".", " "]
        )
        docs = text_splitter.split_documents(documents)

        print(f"Đã chia thành {len(docs)} đoạn văn nhỏ")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)

        vectorstore.save_local(FAISS_INDEX_PATH)
        print("Đã lưu vector store!")

    return vectorstore


system_prompt = """
Bạn là chuyên gia tuyển dụng (HR Expert).
Nhiệm vụ: trả lời câu hỏi chỉ dựa trên nội dung của CV được cung cấp.
Nếu thông tin không có trong CV, hãy nói: "Thông tin này không xuất hiện trong CV."

Yêu cầu:
- Trả lời ngắn gọn, rõ ràng.
- Dùng bullet nếu cần.
- Ưu tiên phân tích kỹ năng, kinh nghiệm, level, phù hợp JD...
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + """

==== CV Content ====
{context}
====================

Câu hỏi: {question}
"""
)


# ===========================
# INIT MODEL & RETRIEVER
# ===========================
print("🔧 Khởi tạo embeddings + FAISS...")
vectorstore = create_or_load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.5}
)

print("🔧 Khởi tạo LLM...")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_output_tokens=1024)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

print("🔥 RAG chatbot sẵn sàng!")


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
    answer = clean_markdown(raw_answer)  # ÁP DỤNG CHUẨN HÓA

    sources = []
    for doc in result["source_documents"]:
        src = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", "N/A")
        sources.append({
            "file": src,
            "page": page
        })

    return jsonify({
        "answer": answer,
        "sources": sources
    }), 200


@app.route("/", methods=["GET"])
def home():
    return "RAG Chatbot Flask Backend is running!", 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
