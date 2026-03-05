import ollama

def generate_answer(query: str, chunks: list[str]):
    # Join chunks into one context block
    context = "\n\n".join(chunks)
    
    # Stuff context + query into a single prompt
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
"""
    
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    for chunk in response:
        yield chunk.message.content