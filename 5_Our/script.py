import os
import sys

# 1. Extract PDF
print("Memulai ekstraksi PDF...")
try:
    import fitz  # PyMuPDF
    doc = fitz.open("kerangka.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    with open("kerangka.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Berhasil mengekstrak kerangka.pdf ke kerangka.txt menggunakan PyMuPDF.")
except ImportError:
    try:
        import PyPDF2
        with open("kerangka.pdf", "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        with open("kerangka.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Berhasil mengekstrak kerangka.pdf ke kerangka.txt menggunakan PyPDF2.")
    except ImportError:
        print("Error: Library PyMuPDF atau PyPDF2 tidak ditemukan. Jalankan: pip install PyMuPDF PyPDF2")

# 2. Download Kaggle Dataset
print("\nMemulai download dataset dari Kaggle...")
try:
    import kagglehub
    path = kagglehub.dataset_download("yenroyenro/harga-rumah-di-dki-jakarta-2024")
    print("Berhasil download! Path to dataset files:", path)
except ImportError:
    print("Error: Library kagglehub tidak ditemukan. Jalankan: pip install kagglehub")
