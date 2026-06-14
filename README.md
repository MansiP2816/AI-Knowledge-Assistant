AI Knowledge Assistant

A simple Retrieval-Augmented Generation (RAG) application built using:

- Python
- OpenAI
- FAISS
- Sentence Transformers

Features

- PDF document ingestion
- Text chunking
- Vector embeddings
- Semantic search
- GPT-powered answers

Installation

pip install -r requirements.txt

Run

python main.py

Architecture

PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Semantic Search
 ↓
GPT
 ↓
Answer
