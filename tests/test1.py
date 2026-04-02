from core.classifier import classify_answer

test_queries = [
    "What is SVM?",
    "Explain how SVM works in detail",
    "Generate 5 quiz questions on SVM",
    "Define kernel trick",
    "Give me a detailed explanation of margin",
    "Create MCQs on support vectors"
]

for query in test_queries:
    label = classify_answer(query)
    print(f"{label.upper():15} ← {query}")