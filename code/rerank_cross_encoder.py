"""Reranking des candidats hybrid_retrieval (+RRF) avec un cross-encoder Hugging Face,
en alternative au reranking par LLM de rerank_LLM.py. Même signature que
rerank_documents(query, documents, top_n) pour rester interchangeable dans retrieval.py."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQUENCE_LENGTH = 512  # longueur maximale des séquences
BATCH_SIZE = 20 # car on a 20 chunks candidats

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
_model.to(_device)
_model.eval()


def rerank_documents(query, documents, top_n=5, batch_size=BATCH_SIZE):
    if not documents:
        return []

    scored_documents = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            pairs = [(query, document["document"]) for document in batch]
            encoded = _tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            ).to(_device)
            # Le modèle ms-marco renvoie un seul logit de pertinence par paire (query, document).
            logits = _model(**encoded).logits.view(-1)
            scores = torch.sigmoid(logits).tolist()

            for document, score in zip(batch, scores):
                scored_document = dict(document)
                scored_document["rerank_score"] = float(score)
                scored_documents.append(scored_document)

    return sorted(
        scored_documents,
        key=lambda document: document["rerank_score"],
        reverse=True,
    )[:top_n]


def main():
    import chromadb

    from retrieval import CHROMA_PATH, COLLECTION_NAME, RRF_CANDIDATES, embed_question, hybrid_retrieval

    question = "quels sont les secteurs d'activité de milliman ?"
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query_embedding = embed_question(question)
    candidates = hybrid_retrieval(collection, question, query_embedding, n_results=RRF_CANDIDATES)
    results = rerank_documents(question, candidates, top_n=5)

    print(f"Question : {question}\n")
    for result in results:
        meta = result["metadata"]
        print(f"Score cross-encoder: {result['rerank_score']:.4f}")
        print(f"Score hybride: {result['hybrid_score']:.4f}")
        print(f"Titre: {meta.get('title')}")
        print(result["document"][:600])
        print("-" * 80)


if __name__ == "__main__":
    main()
