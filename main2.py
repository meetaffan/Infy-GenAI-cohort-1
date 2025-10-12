import os
import re
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------------------
# 🔹 STEP 1 — Detailed Cleaning
# ----------------------------
def clean_text(text: str) -> str:
    """Comprehensive text normalization and cleanup."""
    
    # 1️⃣ Encoding normalization
    text = text.encode("utf-8", "ignore").decode()

    # 2️⃣ Remove HTML artifacts (if any)
    text = re.sub(r"<[^>]+>", " ", text)

    # 3️⃣ Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|@\S+", " ", text)

    # 4️⃣ Replace fancy quotes/dashes with standard ones
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")

    # 5️⃣ Replace bullets and numbered list markers with hyphens
    text = re.sub(r"[\u2022•◦▪‣]", "-", text)
    text = re.sub(r"(\n\s*\d+[\).])", "\n-", text)

    # 6️⃣ Remove page headers/footers like "Page 3 of 20" or "Page 7"
    text = re.sub(r"Page\s*\d+(\s*of\s*\d+)?", " ", text, flags=re.IGNORECASE)

    # 7️⃣ Fix wrapped lines (where lines are broken mid-sentence)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 8️⃣ Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # 9️⃣ Remove unwanted characters (non-printables)
    text = re.sub(r"[^a-zA-Z0-9.,;:?!()\-\n /%]", "", text)

    # 🔟 Normalize punctuation (e.g., “!!!” → “!”)
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # 11️⃣ Normalize multiple blank lines → one
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 12️⃣ Strip leading/trailing whitespace
    text = text.strip()

    return text


# ----------------------------
# 🔹 STEP 2 — File Loader
# ----------------------------
def load_file(file_path: str):
    """Auto-detect file type and load text."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".html", ".htm"]:
        loader = UnstructuredHTMLLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"✅ Loading {ext.upper()} file: {file_path}")
    return loader.load()


# ----------------------------
# 🔹 STEP 3 — Preprocess Docs
# ----------------------------
def preprocess_documents(docs):
    """Clean, normalize, and split into chunks."""
    print("🧽 Cleaning and chunking text...")

    for d in docs:
        d.page_content = clean_text(d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


# ----------------------------
# 🔹 STEP 4 — Save Output
# ----------------------------
def save_to_markdown(chunks, output_dir="outputs"):
    """Save cleaned chunks to timestamped markdown file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"processed_{timestamp}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"### Chunk {i}\n\n")
            f.write(chunk.page_content.strip() + "\n\n---\n\n")

    print(f"💾 Cleaned output saved to: {output_path}")


# ----------------------------
# 🔹 STEP 5 — Main Entry
# ----------------------------
def main():
    file_path = input("📂 Enter path of the file to preprocess: ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found! Please check the path.")
        return

    docs = load_file(file_path)
    chunks = preprocess_documents(docs)
    save_to_markdown(chunks)


if __name__ == "__main__":
    main()
