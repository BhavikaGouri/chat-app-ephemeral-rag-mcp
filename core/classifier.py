from transformers import pipeline

_classifier = None

def load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    return _classifier

LABELS = [
    "retrieve a specific fact or definition",
    "explain in detail with context",
    "generate or create new content"
]

LABEL_MAP = {
    "retrieve a specific fact or definition": "search",
    "explain in detail with context": "explanation",
    "generate or create new content": "generation"
}

def classify_answer(query: str) -> str:
    classifier = load_classifier()
    result = classifier(query, candidate_labels=LABELS)
    return LABEL_MAP.get(result["labels"][0], "unknown")