import os
import re
import unicodedata
from collections import Counter
from math import log

import chromadb
from openai import OpenAI

from .rerank_LLM import rerank_documents # Importation de la fonction de reranking des documents par le LLM, qu'on peut changer plus tard si d'autres méthodes de reranking sont meilleures


QUESTION = "quels sont les secteurs d'activité de milliman ?"
COLLECTION_NAME = "milliman_chunks"
CHROMA_PATH = "chroma_db"
N_RESULTS = 5
RRF_CANDIDATES = 20
RRF_K = 60
RERANK_MODEL = "gpt-4o"
# Ces mots fréquents sont ignorés pour limiter le bruit dans le score lexical.
STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "for", "from", "il", "ils", "in", "la", "le",
    "les", "leur", "mais", "of", "on", "or", "par", "pas", "pour", "que",
    "quel", "quelle", "quelles", "quels", "qui", "sont", "sur", "the", "un",
    "une", "what", "which", "with",
}

client = OpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://openai-paris-rnd.openai.azure.com/openai/v1",
)

# Fonction pour obtenir l'embedding d'une question (pour faire la recherche sémantique plus tard)
def embed_question(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding

# Fonction pour tokeniser un texte (pour utiliser le BM25)
def tokenize(text):
    # Normaliser les accents et extraire les mots
    normalized_text = unicodedata.normalize("NFKD", text)
    normalized_text = "".join(
        character
        for character in normalized_text
        if not unicodedata.combining(character)
    )
    tokens = re.findall(r"[a-z0-9]+", normalized_text.lower())
    return [token for token in tokens if token not in STOPWORDS]

# Fonction pour normaliser les scores (pour combiner BM25 et recherche vectorielle)
def normalize_scores(scores):
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {doc_id: 1.0 for doc_id in scores}
    return {
        doc_id: (score - minimum) / (maximum - minimum)
        for doc_id, score in scores.items()
    }

# Fonction pour calculer les scores BM25 d'une liste de documents par rapport à une question
def bm25_scores(documents, question, k1=1.5, b=0.75):
    tokenized_documents = [tokenize(document) for document in documents]
    query_tokens = tokenize(question)
    document_count = len(tokenized_documents)
    average_length = sum(map(len, tokenized_documents)) / max(1, document_count)
    document_frequencies = Counter(
        token
        for document in tokenized_documents
        for token in set(document)
    )
    scores = []

    for document in tokenized_documents:
        term_frequencies = Counter(document)
        document_length = len(document)
        score = 0.0
        for token in query_tokens:
            if token not in term_frequencies:
                continue
            frequency = term_frequencies[token]
            document_frequency = document_frequencies[token]
            idf = log(
                1.0 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * document_length / max(1, average_length)
            )
            score += idf * frequency * (k1 + 1.0) / denominator
        scores.append(score)

    return scores

# Fonction pour effectuer une recherche hybride combinant BM25 et recherche vectorielle
def hybrid_retrieval(collection, question, query_embedding, n_results=5, where=None):
    # Charge les textes pour calculer BM25 et les métadonnées à renvoyer avec les résultats.
    corpus = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    documents = corpus["documents"]
    document_ids = [str(doc_id) for doc_id in corpus["ids"]]
    metadatas = corpus["metadatas"]

    # Recherche vectorielle : Chroma renvoie les candidats les plus proches de la question.
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(RRF_CANDIDATES, len(document_ids)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    semantic_scores = {
        str(doc_id): 1.0 / (1.0 + distance)
        for doc_id, distance in zip(
            semantic_results["ids"][0], semantic_results["distances"][0]
        )
    }

    # Recherche lexicale BM25 calculée sur les mêmes documents du corpus.
    lexical_scores = {
        doc_id: score
        for doc_id, score in zip(document_ids, bm25_scores(documents, question))
    }

    semantic_scores = normalize_scores(semantic_scores)
    lexical_scores = normalize_scores(lexical_scores)

    semantic_ranked_ids = [
        str(doc_id)
        for doc_id in semantic_results["ids"][0]
    ]
    lexical_ranked_ids = sorted(
        lexical_scores,
        key=lexical_scores.get,
        reverse=True,
    )[:RRF_CANDIDATES]
    # RRF combine les rangs des deux recherches (1ère étape de reranking)
    rrf_scores = Counter()
    for rank, doc_id in enumerate(semantic_ranked_ids, start=1):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(lexical_ranked_ids, start=1):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank)

    combined_scores = {
        doc_id: rrf_scores.get(doc_id, 0.0)
        for doc_id in set(semantic_ranked_ids) | set(lexical_ranked_ids)
    }

    ranked_ids = sorted(
        combined_scores,
        key=combined_scores.get,
        reverse=True,
    )[:n_results]
    metadata_by_id = dict(zip(document_ids, metadatas))
    document_by_id = dict(zip(document_ids, documents))

    return [
        {
            "id": doc_id,
            "document": document_by_id[doc_id],
            "metadata": metadata_by_id[doc_id],
            "semantic_score": semantic_scores.get(doc_id, 0.0),
            "bm25_score": lexical_scores.get(doc_id, 0.0),
            "hybrid_score": combined_scores[doc_id],
        }
        for doc_id in ranked_ids
    ]


def main():
    # Le flux complet : embedding de la question, retrieval hybride, puis reranking LLM.
    query_embedding = embed_question(QUESTION)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    candidates = hybrid_retrieval(
        collection,
        QUESTION,
        query_embedding,
        n_results=RRF_CANDIDATES,
    )
    results = rerank_documents(QUESTION, candidates, top_n=N_RESULTS) #appel de la fonction de reranking par le LLM

    print(f"Question : {QUESTION}")
    print("\nTop résultats :\n")

    for result in results:
        meta = result["metadata"]
        print(f"Score reranking: {result['rerank_score']:.1f}/10")
        print(f"Score hybride: {result['hybrid_score']:.4f}")
        print(f"Score sémantique: {result['semantic_score']:.4f}")
        print(f"Score BM25: {result['bm25_score']:.4f}")
        print(f"Titre: {meta.get('title')}")
        print(f"URL: {meta.get('url')}")
        print(result["document"][:600])
        print("-" * 80)


if __name__ == "__main__":
    main()
