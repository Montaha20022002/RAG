import json
import os
import time
from math import log2
from pathlib import Path
import chromadb
import mlflow
from openai import OpenAI
from retrieval import RRF_CANDIDATES, hybrid_retrieval, rerank_documents

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
GROUND_TRUTH_PATH = ROOT / "data" / "fichiers json" / "GROUND_TRUTH_final.json"
COLLECTION_NAME = "milliman_chunks"
N_RESULTS = 10
K_VALUES = [1, 3, 5, 10]
# Mettre une année ici pour limiter l'évaluation aux questions liées à cette année.
PUBLICATION_YEAR = None
OUTPUT_PATH = ROOT / "data" / "fichiers json" / "groundtruth_topic_retrieval_metrics.json"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "RAG Retrieval Evaluation"

client = OpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://openai-paris-rnd.openai.azure.com/openai/v1",
)

def embed_question(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding


def load_ground_truth(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ground_truth.json must contain a list of questions")
    return data


def get_retrieval_relevant_ids(item):
    """Return all chunks that can legitimately support the answer."""
    relevant_ids = []
    seen_ids = set()
    for label in ("necessary_chunk_ids", "relevant_chunk_ids", "redundant_chunk_ids"):
        for chunk_id in item.get(label, []):
            normalized_id = str(chunk_id)
            if normalized_id not in seen_ids:
                seen_ids.add(normalized_id)
                relevant_ids.append(normalized_id)
    return relevant_ids


def get_unique_chunk_ids(item):
    """Use new unique labels while remaining compatible with older ground-truth files."""
    return [
        str(chunk_id)
        for chunk_id in item.get("unique_chunk_ids", item.get("necessary_chunk_ids", []))
    ]


def compute_recall_precision(relevant_ids, retrieved_ids, k):
    # Un hit inclut tout chunk pertinent, qu'il soit unique ou overlapping.
    relevant_set = {str(x) for x in relevant_ids}
    retrieved_k = [str(x) for x in retrieved_ids[:k]]
    hits = [x for x in retrieved_k if x in relevant_set]
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
    precision = len(hits) / max(1, len(retrieved_k))
    return recall, precision, bool(hits)


def compute_mrr(relevant_ids, retrieved_ids):
    # Mesure la position du premier chunk pertinent retrouvé dans le classement.
    relevant_set = {str(x) for x in relevant_ids}
    for rank, item in enumerate(retrieved_ids, start=1):
        if str(item) in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg(relevant_ids, retrieved_ids, k):
    # Récompense les chunks pertinents placés tôt dans les k premiers résultats.
    relevant_set = {str(x) for x in relevant_ids}
    dcg = 0.0
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if str(item) in relevant_set:
            dcg += 1.0 / log2(idx + 1)

    ideal_count = min(len(relevant_ids), k)
    idcg = 0.0
    for idx in range(1, ideal_count + 1):
        idcg += 1.0 / log2(idx + 1)

    return dcg / idcg if idcg > 0 else 0.0


def get_question_metadata(collection, records):
    # Récupère la discipline et l'année des chunks portant une information unique.
    chunk_ids = sorted({
        str(chunk_id)
        for item in records
        for chunk_id in get_unique_chunk_ids(item)
    })
    chunk_data = collection.get(ids=chunk_ids, include=["metadatas"])
    chunk_metadata = {
        str(chunk_id): metadata
        for chunk_id, metadata in zip(chunk_data["ids"], chunk_data["metadatas"])
        if isinstance(metadata, dict)
    }

    question_metadata = {}
    for index, item in enumerate(records):
        disciplines = {
            str(chunk_metadata[chunk_id].get("discipline_base")).strip()
            for chunk_id in get_unique_chunk_ids(item)
            if chunk_id in chunk_metadata and chunk_metadata[chunk_id].get("discipline_base")
        }
        years = {
            int(chunk_metadata[chunk_id]["year"])
            for chunk_id in get_unique_chunk_ids(item)
            if chunk_id in chunk_metadata and chunk_metadata[chunk_id].get("year") is not None
        }
        question_metadata[index] = {
            "discipline": (
                next(iter(disciplines))
                if len(disciplines) == 1
                else "multiple_disciplines"
                if disciplines
                else "unknown"
            ),
            "years": years,
        }
    return question_metadata


def evaluate_questions(collection, records, question_metadata):

    summary = {k: {"recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "ndcg": 0.0} for k in K_VALUES}
    mrr_total = 0.0
    hit10_total = 0.0
    ndcg10_total = 0.0
    latency_total = 0.0
    embedding_latency_total = 0.0
    retrieval_latency_total = 0.0
    rerank_latency_total = 0.0
    evaluated = 0
    total_questions = len(records)

    per_question = []

    for index, item in enumerate(records):
        processed = index + 1
        if processed == 1 or processed % 10 == 0 or processed == total_questions:
            progress = processed / max(1, total_questions) * 100
            print(
                f"\rProgression : {processed}/{total_questions} "
                f"({progress:.1f} %) - questions évaluées : {evaluated}",
                end="",
                flush=True,
            )
        question = str(item.get("question", "")).strip()
        topic = item.get("topic")
        difficulty = item.get("difficulty")
        unique_ids = get_unique_chunk_ids(item)
        relevant_ids = get_retrieval_relevant_ids(item)
        discipline = question_metadata[index]["discipline"]
        publication_years = question_metadata[index]["years"]
        # Ignore les questions hors filtre, vides ou sans chunk de référence.
        if PUBLICATION_YEAR is not None and PUBLICATION_YEAR not in publication_years:
            continue
        if not question:
            continue
        if not unique_ids or not relevant_ids:
            continue

        evaluated += 1

        # Chronomètre séparément l'embedding, le retrieval hybride et le reranking LLM.
        embedding_start = time.perf_counter()
        embedding = embed_question(question)
        embedding_latency = time.perf_counter() - embedding_start

        retrieval_start = time.perf_counter()
        candidates = hybrid_retrieval(
            collection,
            question,
            embedding,
            n_results=RRF_CANDIDATES,
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        rerank_start = time.perf_counter()
        retrieved_results = rerank_documents(question, candidates, top_n=N_RESULTS)
        rerank_latency = time.perf_counter() - rerank_start

        # La latence totale correspond à la somme des trois étapes chronométrées.
        total_latency = embedding_latency + retrieval_latency + rerank_latency
        latency_total += total_latency
        embedding_latency_total += embedding_latency
        retrieval_latency_total += retrieval_latency
        rerank_latency_total += rerank_latency

        retrieved_ids = [str(result["id"]) for result in retrieved_results]

        per_question_metrics = {
            "question": question,
            "topic": topic,
            "discipline": discipline,
            "publication_years": sorted(publication_years),
            "difficulty": difficulty,
            "unique_chunk_ids": unique_ids,
            "retrieval_relevant_chunk_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "latency_seconds": total_latency,
            "embedding_latency_seconds": embedding_latency,
            "retrieval_latency_seconds": retrieval_latency,
            "rerank_latency_seconds": rerank_latency,
        }
        # Calcule les métriques pour plusieurs tailles de résultats (top 1, 3, 5 et 10).
        for k in K_VALUES:
            recall, precision, hit = compute_recall_precision(relevant_ids, retrieved_ids, k)
            summary[k]["recall"] += recall
            summary[k]["precision"] += precision
            summary[k]["hit_rate"] += 1.0 if hit else 0.0
            summary[k]["ndcg"] += compute_ndcg(relevant_ids, retrieved_ids, k)
            per_question_metrics[f"recall@{k}"] = recall
            per_question_metrics[f"precision@{k}"] = precision
            per_question_metrics[f"hit@{k}"] = 1.0 if hit else 0.0
            per_question_metrics[f"ndcg@{k}"] = compute_ndcg(relevant_ids, retrieved_ids, k)

        mrr_total += compute_mrr(relevant_ids, retrieved_ids)
        hit10_total += 1.0 if any(str(x) in set(relevant_ids) for x in retrieved_ids[:10]) else 0.0
        ndcg10_total += compute_ndcg(relevant_ids, retrieved_ids, 10)
        per_question_metrics["mrr"] = compute_mrr(relevant_ids, retrieved_ids)
        per_question_metrics["hit@10"] = 1.0 if any(str(x) in set(relevant_ids) for x in retrieved_ids[:10]) else 0.0
        per_question_metrics["ndcg@10"] = compute_ndcg(relevant_ids, retrieved_ids, 10)
        per_question.append(per_question_metrics)

    print()

    if evaluated == 0:
        raise ValueError(f"Aucune question trouvée pour l'année '{PUBLICATION_YEAR}'.")

    avg_summary = {
        k: {
            "recall": summary[k]["recall"] / evaluated,
            "precision": summary[k]["precision"] / evaluated,
            "hit_rate": summary[k]["hit_rate"] / evaluated,
            "ndcg": summary[k]["ndcg"] / evaluated,
        }
        for k in K_VALUES
    }

    final = {
        "num_questions": evaluated,
        "metrics": {
            "Recall@1": avg_summary[1]["recall"],
            "Precision@1": avg_summary[1]["precision"],
            "HitRate@1": avg_summary[1]["hit_rate"],
            "NDCG@1": avg_summary[1]["ndcg"],
            "Recall@3": avg_summary[3]["recall"],
            "Precision@3": avg_summary[3]["precision"],
            "HitRate@3": avg_summary[3]["hit_rate"],
            "NDCG@3": avg_summary[3]["ndcg"],
            "Recall@5": avg_summary[5]["recall"],
            "Precision@5": avg_summary[5]["precision"],
            "HitRate@5": avg_summary[5]["hit_rate"],
            "NDCG@5": avg_summary[5]["ndcg"],
            "Recall@10": avg_summary[10]["recall"],
            "Precision@10": avg_summary[10]["precision"],
            "HitRate@10": avg_summary[10]["hit_rate"],
            "NDCG@10": avg_summary[10]["ndcg"],
            "MRR": mrr_total / evaluated,
            "Avg_latency_seconds": latency_total / evaluated,
            "Avg_embedding_latency_seconds": embedding_latency_total / evaluated,
            "Avg_retrieval_latency_seconds": retrieval_latency_total / evaluated,
            "Avg_rerank_latency_seconds": rerank_latency_total / evaluated,
        },
        "per_question": per_question,
    }

    return final


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
    records = load_ground_truth(GROUND_TRUTH_PATH)
    question_metadata = get_question_metadata(collection, records)

    with mlflow.start_run(run_name="hybrid-retrieval-gpt4o-rerank"):
        mlflow.log_params(
            {
                "collection": COLLECTION_NAME,
                "embedding_model": "text-embedding-3-large",
                "rerank_model": "gpt-4o",
                "rrf_candidates": RRF_CANDIDATES,
                "n_results": N_RESULTS,
                "k_values": ",".join(map(str, K_VALUES)),
                "publication_year": PUBLICATION_YEAR or "all",
            }
        )

        results = evaluate_questions(collection, records, question_metadata)

        # Enregistre le résumé global et les résultats détaillés de chaque question.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        mlflow.log_param("num_questions", results["num_questions"])
        mlflow.log_metrics(
            {name: float(value) for name, value in results["metrics"].items()}
        )

    print(f"Année de publication filtrée : {PUBLICATION_YEAR or 'toutes'}")
    print(f"Résultats exportés dans : {OUTPUT_PATH}")
    metrics = results["metrics"]
    print(f"\nToutes les disciplines ({results['num_questions']} questions)")
    print("=" * 90)
    print("{:<6} | {:>12} | {:>14} | {:>12} | {:>10}".format(
        "K", "Recall@K", "Precision@K", "HitRate@K", "NDCG@K"
    ))
    print("-" * 90)
    for k in K_VALUES:
        print("{:<6} | {:>12.4f} | {:>14.4f} | {:>12.4f} | {:>10.4f}".format(
            k,
            metrics[f"Recall@{k}"],
            metrics[f"Precision@{k}"],
            metrics[f"HitRate@{k}"],
            metrics[f"NDCG@{k}"],
        ))
    print("-" * 90)
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"Avg latency total: {metrics['Avg_latency_seconds']:.3f} s")
    print(f"Avg embedding latency: {metrics['Avg_embedding_latency_seconds']:.3f} s")
    print(f"Avg retrieval latency: {metrics['Avg_retrieval_latency_seconds']:.3f} s")
    print(f"Avg rerank latency: {metrics['Avg_rerank_latency_seconds']:.3f} s")
    print(f"Avg latency total: {metrics['Avg_latency_seconds']:.3f} s")
    print(f"Avg embedding latency: {metrics['Avg_embedding_latency_seconds']:.3f} s")
    print(f"Avg retrieval latency: {metrics['Avg_retrieval_latency_seconds']:.3f} s")
    print(f"Avg rerank latency: {metrics['Avg_rerank_latency_seconds']:.3f} s")


if __name__ == "__main__":
    main()
