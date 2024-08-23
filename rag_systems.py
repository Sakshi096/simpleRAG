
from data import documents
from retrieval import embed_documents, create_faiss_index, retrieve
from generation import generate

# Embed the documents
doc_embeddings = embed_documents(documents)

# Create FAISS index
index = create_faiss_index(doc_embeddings)

# Function to perform Retrieval-Augmented Generation
def rag(query):
    retrieved_docs, distances = retrieve(query, index, documents)
    generated_response = generate(retrieved_docs)
    return {
        'query': query,
        'retrieved_docs': retrieved_docs,
        'generated_response': generated_response
    }

# Test the RAG system with some queries
if __name__ == "__main__":
    queries = [
        "Tell me about transformers.",
        "What is Hugging Face?",
        "How does RAG work?"
    ]

    for query in queries:
        result = rag(query)
        print(f"Query: {result['query']}")
        print(f"Retrieved Documents: {result['retrieved_docs']}")
        print(f"Generated Response: {result['generated_response']}\n")
