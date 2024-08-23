# Retrieval-Augmented Generation (RAG) System

This project demonstrates a basic Retrieval-Augmented Generation (RAG) system using open-source libraries like Hugging Face's Transformers, Sentence Transformers, and FAISS for efficient document retrieval and text generation.

## Project Structure

```plaintext
rag_system/
│
├── data.py           # Contains the document data
├── retrieval.py      # Handles document embedding and retrieval logic
├── generation.py     # Manages text generation based on retrieved documents
├── rag_system.py     # Integrates the retrieval and generation components
├── README.md         # Project documentation
└── requirements.txt  # List of dependencies
