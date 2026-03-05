from sentence_transformers import SentenceTransformer
import chromadb

#Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_ephemeral_index(chunks):
    # Initialize the embedding model

    content = [chunk.page_content for chunk in chunks]

    # Generate embeddings for the content
    embeddings = model.encode(content).tolist()

    # Create a ChromaDB client and collection
    client = chromadb.Client()
    try:
        client.delete_collection("session_index")
    except:
        pass

    collection = client.create_collection(name="session_index")

    # Add documents and their embeddings to the collection
    collection.add(
        documents=content,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(content))]
    )
    return collection


def retrieve_relevant_chunks(query, collection, top_k=3):
    # Generate embedding for the query
    query_embedding = model.encode([query]).tolist()

    # Retrieve relevant chunks based on cosine similarity
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]