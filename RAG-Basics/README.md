# RAG System - Document Retrieval & Analysis

A Retrieval-Augmented Generation (RAG) system built with LangChain, SentenceTransformers, ChromaDB, and Groq LLM for semantic document search and question answering.

## Features

- **PDF Document Loading**: Extract text from multiple PDF files using PyPDF / PyMuPDF
- **Text Chunking**: Intelligent document splitting with overlap for context preservation
- **Embeddings**: Generate semantic embeddings using `all-MiniLM-L6-v2` model
- **Vector Store**: Persistent vector database using ChromaDB
- **Semantic Search**: Retrieve relevant documents based on semantic similarity
- **LLM Integration**: Answer generation using Groq API (llama-3.1-8b-instant)
- **RAG Pipeline**: End-to-end retrieval + generation pipeline for document-based Q&A

## Project Structure

```
RAG-Basics/
├── data/
│   ├── pdf_files/       # PDF documents to process
│   ├── text_files/      # Text documents
│   ├── vector_store/    # ChromaDB persistent storage
│   └── README.md
└── notebook/
    └── document.ipynb   # Main Jupyter notebook
```

## Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd RAG-Basics
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Usage

Open `notebook/document.ipynb` and run the cells in order.

### RAG Pipeline

```python
# Initialize components
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_retriever = RAGRetrieval(vectorstore, embedding_manager)

# Load and process PDFs
documents = process_all_pdfs("../data")
chunks = split_documents(documents)

# Generate embeddings and store
embeddings = embedding_manager.generate_embeddings(texts)
vectorstore.add_documents(chunks, embeddings)

# Initialize LLM
from langchain_groq import ChatGroq
import os
llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model="llama-3.1-8b-instant")

# Ask a question
answer = rag_simple("What is FGSM?", rag_retriever, llm)
print(answer)
```

## Key Components

| Component | Description |
|-----------|-------------|
| `EmbeddingManager` | Generates semantic embeddings using SentenceTransformers |
| `VectorStore` | Manages ChromaDB integration with batched insertion |
| `RAGRetrieval` | Retrieves relevant chunks using semantic similarity search |
| `rag_simple` | End-to-end RAG function: retrieves context and generates an answer |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Number of chunks to retrieve |
| `score_threshold` | 0.3 | Minimum similarity score (0–1) |
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `batch_size` | 4000 | ChromaDB insertion batch size |

## Dependencies

See `requirements.txt`. Key libraries:

- **LangChain** — document loading, text splitting, LLM abstraction
- **SentenceTransformers** — local semantic embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** — persistent vector database
- **PyPDF / PyMuPDF** — PDF text extraction
- **langchain-groq** — Groq LLM integration
- **python-dotenv** — environment variable management

## Notes

- Similarity score uses L2 distance: `score = 1 / (1 + distance)`
- UTF-8 encoding is cleaned on ingestion to handle special characters in PDFs
- ChromaDB batching is capped at 4000 items per call to stay within limits

## Author

Karmabir Chakraborty
