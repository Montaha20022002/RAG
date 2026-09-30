"""Génère des questions par topic/difficulté à partir des chunks de Chroma.
Ce script s'arrête à la génération des requêtes : le retrieval, la génération
de réponse et l'évaluation (DeepEval / LLM-as-a-judge) sont faits dans une
étape séparée, à partir du fichier de sortie produit ici."""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from openai import AzureOpenAI


ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "milliman_chunks"
OUTPUT_PATH = ROOT / "data" / "fichiers json" / "generated_queries.json"

AZURE_ENDPOINT = "https://openai-paris-rnd-swedencentral.openai.azure.com/"
API_VERSION = "2024-12-01-preview"
GENERATION_MODEL = "gpt-4.1"
QUERIES_PER_TOPIC_DIFFICULTY = 2
MIN_CHUNK_WORDS = 25
DIFFICULTIES = ("easy", "medium", "hard")

# Chaque niveau fixe combien de passages source inspirent la question, le style attendu,
# et à quelle distance sémantique (par rapport au chunk "seed") les voisins sont choisis :
# plus la difficulté augmente, plus les chunks combinés sont sémantiquement éloignés entre eux.
DIFFICULTY_PROFILES = {
    "easy": {
        "sample_chunks": 1,
        "distance_fractions": [],
        "instructions": "Une question simple et directe, répondable à partir d'un seul passage.",
    },
    "medium": {
        "sample_chunks": 2,
        "distance_fractions": [0.5],
        "instructions": "Une question qui nécessite de combiner deux passages liés du même topic.",
    },
    "hard": {
        "sample_chunks": 3,
        "distance_fractions": [0.6, 0.9],
        "instructions": (
            "Une question complexe qui nécessite de synthétiser plusieurs passages, "
            "avec une reformulation éloignée du vocabulaire des textes source."
        ),
    },
}

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)


def load_topic_chunks(collection, limit=50000):
    """Regroupe les chunks nettoyés par topic_name, avec leurs embeddings, comme dans ground_truth_2.0."""
    items = collection.get(include=["documents", "metadatas", "embeddings"], limit=limit)
    topics = defaultdict(list)
    embeddings = items.get("embeddings")
    for index, (chunk_id, document, metadata) in enumerate(
        zip(items["ids"], items["documents"], items.get("metadatas") or [])
    ):
        if not isinstance(metadata, dict) or not metadata.get("topic_name"):
            continue
        text = " ".join(str(document).split())
        if len(text.split()) < MIN_CHUNK_WORDS:
            continue
        embedding = embeddings[index] if embeddings is not None else None
        topics[str(metadata["topic_name"])].append(
            {
                "id": str(chunk_id),
                "text": text,
                "title": metadata.get("title", ""),
                "embedding": embedding,
            }
        )
    return topics


def cosine_similarity(vector_a, vector_b):
    if vector_a is None or vector_b is None:
        return 0.0
    vector_a = np.asarray(vector_a, dtype=np.float32)
    vector_b = np.asarray(vector_b, dtype=np.float32)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def select_chunks_by_difficulty(chunks, difficulty, rng):
    """Choisit un chunk seed puis des voisins de plus en plus éloignés selon la difficulté."""
    profile = DIFFICULTY_PROFILES[difficulty]
    group_size = profile["sample_chunks"]
    if len(chunks) < group_size:
        return None

    seed = rng.choice(chunks)
    if group_size == 1:
        return [seed]

    if any(chunk.get("embedding") is None for chunk in chunks):
        # Sans embeddings, on retombe sur un tirage aléatoire pour rester robuste.
        others = rng.sample([chunk for chunk in chunks if chunk["id"] != seed["id"]], group_size - 1)
        return [seed, *others]

    neighbors = sorted(
        (
            (cosine_similarity(seed["embedding"], chunk["embedding"]), chunk)
            for chunk in chunks
            if chunk["id"] != seed["id"]
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    needed = group_size - 1
    if len(neighbors) < needed:
        return None

    # Une fraction proche de 0 = voisin très similaire (facile), proche de 1 = très éloigné (difficile).
    indices = sorted(
        {min(len(neighbors) - 1, round(fraction * (len(neighbors) - 1))) for fraction in profile["distance_fractions"]}
    )
    selected = [neighbors[index][1] for index in indices]
    if len(selected) < needed:
        selected_ids = {chunk["id"] for chunk in selected}
        selected.extend(item[1] for item in neighbors if item[1]["id"] not in selected_ids)
    selected = selected[:needed]

    group = [seed, *selected]
    return group if len({chunk["id"] for chunk in group}) == group_size else None


def build_query_prompt(topic, difficulty, chunks):
    profile = DIFFICULTY_PROFILES[difficulty]
    context_block = "\n\n---\n\n".join(
        f"TITLE: {chunk['title']}\nTEXT:\n{chunk['text']}" for chunk in chunks
    )
    return (
        f"Tu génères une question réaliste qu'un utilisateur poserait à un système de RAG, "
        f"sur le topic \"{topic}\".\n"
        f"Niveau de difficulté demandé : {difficulty}. {profile['instructions']}\n"
        "La question ne doit pas recopier mot pour mot un titre ou une phrase du texte source.\n"
        "Retourne uniquement la question, sans guillemets ni explication.\n\n"
        f"Extraits de référence :\n{context_block}"
    )


def generate_query(topic, difficulty, chunks):
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Tu génères des questions de benchmark pour évaluer un système de RAG.",
            },
            {"role": "user", "content": build_query_prompt(topic, difficulty, chunks)},
        ],
        temperature=0.8, #on cherche à obtenir des questions variées et réalistes à partir du corpus.
        max_tokens=200,
    )
    return (response.choices[0].message.content or "").strip().strip('"')


def build_dataset(topic_chunks, queries_per_combo, rng):
    records = []
    total_combos = len(topic_chunks) * len(DIFFICULTIES)
    combo_index = 0
    for topic, chunks in topic_chunks.items():
        for difficulty in DIFFICULTIES:
            combo_index += 1
            profile = DIFFICULTY_PROFILES[difficulty]
            if len(chunks) < profile["sample_chunks"]:
                continue
            for _ in range(queries_per_combo):
                sampled_chunks = select_chunks_by_difficulty(chunks, difficulty, rng)
                if not sampled_chunks:
                    print(f"REJECT sélection chunks topic={topic!r} difficulty={difficulty}: pas assez de voisins distincts")
                    continue
                try:
                    question = generate_query(topic, difficulty, sampled_chunks)
                except Exception as exc:
                    print(f"REJECT génération question topic={topic!r} difficulty={difficulty}: {exc}")
                    continue
                if not question:
                    continue

                records.append(
                    {
                        "topic": topic,
                        "difficulty": difficulty,
                        "question": question,
                        "seed_chunk_ids": [chunk["id"] for chunk in sampled_chunks],
                    }
                )
            print(f"Combo {combo_index}/{total_combos} traité : {topic} / {difficulty}")
    return records


def save_json(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def run_pipeline(output_path, queries_per_combo, topics_limit, seed):
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    topic_chunks = load_topic_chunks(collection)
    if not topic_chunks:
        raise ValueError("Aucun topic avec des chunks exploitables dans ChromaDB.")

    if topics_limit:
        selected_topics = sorted(topic_chunks)[:topics_limit]
        topic_chunks = {topic: topic_chunks[topic] for topic in selected_topics}

    rng = random.Random(seed)
    records = build_dataset(topic_chunks, queries_per_combo, rng)
    save_json(records, output_path)
    print(f"Questions générées sauvegardées : {output_path}")
    print(f"Questions générées : {len(records)}")


def main():
    parser = argparse.ArgumentParser(
        description="Génère des questions par topic/difficulté à partir des chunks de Chroma."
    )
    parser.add_argument("--run", action="store_true", help="Confirme les appels API potentiellement nombreux et coûteux.")
    parser.add_argument("--topics-limit", type=int, default=0, help="Limite le nombre de topics traités (0 = tous).")
    parser.add_argument(
        "--queries-per-combo",
        type=int,
        default=QUERIES_PER_TOPIC_DIFFICULTY,
        help="Nombre de questions générées par topic et par niveau de difficulté.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.run:
        print("Aucun appel modèle lancé. Ajouter --run pour confirmer la génération.")
        print(f"Sortie: {args.output}")
        print("Commencer par un pilote: python code/query_chunk_list.py --run --topics-limit 2 --queries-per-combo 1")
        return

    run_pipeline(
        output_path=args.output,
        queries_per_combo=args.queries_per_combo,
        topics_limit=args.topics_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
