import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb


@st.cache_resource  # loads model once, persists across reruns
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()


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



def retrieve_chunks_mmr(query, collection, top_k=5, lambda_param=0.5):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k*2 ,
        include=["documents", "embeddings"]
    )
    if results.get("embeddings") is None or len(results["embeddings"]) == 0:
        return []

    candidate_chunks = results['documents'][0]
    candidate_embeddings = results['embeddings'][0]
    query_vec = query_embedding[0]
    query_similarity = [
    cosine_similarity([query_vec], [emb])[0][0]
    for emb in candidate_embeddings
    ]
    selected = []
    remaining = list(range(len(candidate_embeddings)))
    for _ in range(top_k):
        if not remaining:
            break
        mmr_scores = []
        for idx in remaining:
            relevance = query_similarity[idx]
            diversity = max(
                cosine_similarity([candidate_embeddings[idx]], [candidate_embeddings[i]])[0][0]
                for i in selected
            ) if selected else 0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr_score, idx))
        mmr_scores.sort(reverse=True)
        best_idx = mmr_scores[0][1]
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidate_chunks[i] for i in selected]
