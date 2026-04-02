import ollama


def generate_answer(query: str, chunks: list[str]):
    # Join chunks into one context block
    context = "\n\n".join(chunks)
    # Stuff context + query into a single prompt
    prompt = f"""You are a helpful document assistant.
Use the context below to answer the question.
If the answer is not in the context, answer from general knowledge but say "Based on my general knowledge:".

Context:
{context}

Question: {query}

Answer:"""

    
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    for chunk in response:
        yield chunk.message.content


def generate_answer_with_llm(query: str):
    
    # Stuff context + query into a single prompt
    prompt = f"""You are a helpful assistant. Reply to the user's query based on your general knowledge.At the end of your answer, 
    include a short note about how you arrived at the answer (e.g. "Based on my general knowledge"). Also state that user can send pdf documents for more specific answers.
Question: {query}
Answer:"""

    
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    for chunk in response:
        yield chunk.message.content