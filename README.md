# RAG System - Document Retrieval & Analysis

A Retrieval-Augmented Generation (RAG) system built with LangChain, SentenceTransformers, and ChromaDB for semantic document search and analysis.

## Features

- **PDF Document Loading**: Extract text from multiple PDF files
- **Text Chunking**: Intelligent document splitting with overlap for context preservation
- **Embeddings**: Generate semantic embeddings using `all-MiniLM-L6-v2` model
- **Vector Store**: Persistent vector database using ChromaDB
- **Semantic Search**: Retrieve relevant documents based on semantic similarity
- **RAG Pipeline**: Complete retrieval pipeline for document-based Q&A

## Project Structure

```
├── RAG-Basics/
│   ├── data/
│   │   ├── pdf_files/       # PDF documents to process
│   │   ├── text_files/      # Text documents
│   │   └── vector_store/    # ChromaDB persistent storage
│   └── notebook/
│       └── document.ipynb   # Main Jupyter notebook
├── it_ticket_intelligent_agent/  # Related projects
├── RAG-KrishNaik-1/              # Reference implementations
└── README.md
```

## Installation

1. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install langchain langchain-community pypdf sentence-transformers chromadb numpy
```

## Usage

### Basic RAG Pipeline

```python
# Initialize components
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()

# Load and process documents
documents = loader.load()
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store
embeddings = embedding_manager.generate_embeddings(texts)
vectorstore.add_documents(chunks, embeddings)

# Retrieve documents
rag_retriever = RAGRetrieval(vectorstore, embedding_manager)
results = rag_retriever.retrieve("Query text", top_k=3, score_threshold=0.3)
```

## Key Components

- **EmbeddingManager**: Handles embedding generation using SentenceTransformers
- **VectorStore**: Manages document embeddings and ChromaDB integration with batching support
- **RAGRetrieval**: Retrieval pipeline with semantic search and similarity scoring

## Performance

- **Documents Processed**: 804 pages
- **Chunks Created**: 5,527 document chunks
- **Embedding Dimension**: 384
- **Storage**: Persistent ChromaDB with 4000-item batching

## Configuration

Adjust retrieval parameters:
- `top_k`: Number of results to return (default: 5)
- `score_threshold`: Minimum similarity score (default: 0.3, range: 0-1)
- `batch_size`: ChromaDB batch size (default: 4000)

## Technologies

- **LangChain**: Document loading and text splitting
- **SentenceTransformers**: Semantic embeddings
- **ChromaDB**: Vector database
- **PyPDF**: PDF processing
- **Jupyter**: Interactive development

## Notes

- Uses L2 distance for similarity calculations: `1 / (1 + distance)`
- UTF-8 encoding cleaning applied to handle special characters
- Batching implemented for ChromaDB batch size limits

## Future Enhancements

- [ ] Add LLM integration for answer generation
- [ ] Implement multi-query retrieval strategies
- [ ] Add relevance feedback mechanisms
- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Caching for frequently accessed documents

## Author

Karmabir Chakraborty

## License

MIT
