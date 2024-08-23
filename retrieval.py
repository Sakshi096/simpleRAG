
from sentence_transformers import SentenceTransformer
import faiss

# Load pre-trained sentence transformer model
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the documents (you'll import `documents` from data.py)
def embed_documents(documents):
    return retrieval_model.encode(documents)

# Create FAISS index for similarity search
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve the top-k similar documents
def retrieve(query, index, documents, top_k=2):
    query_embedding = retrieval_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]], distances[0]
