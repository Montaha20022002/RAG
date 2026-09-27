import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from openai import AzureOpenAI


ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "milliman_chunks"
OUTPUT_PATH = ROOT / "data" / "ground_truth.json"

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")

# Chaque niveau de difficulté précise combien de chunks sont proposés et combien doivent vraiment servir à répondre
SIMILARITY_THRESHOLD = 0.8
DIFFICULTY_PROFILES = {
    "easy": {
        "candidate_chunks": 2,
        "min_chunks": 1,
        "max_chunks": 2,
        "combination": "simple",
        "reformulation": "faible",
        "concepts": 1,
        "lexical_gap": "faible",
    },
    "medium": {
        "candidate_chunks": 4,
        "min_chunks": 2,
        "max_chunks": 3,
        "combination": "modérée", 
        "reformulation": "moyenne", 
        "concepts": 2,
        "lexical_gap": "moyen", 

    },
    "hard": {
        "candidate_chunks": 6,
        "min_chunks": 3,
        "max_chunks": 5,
        "combination": "élevée",
        "reformulation": "forte",
        "concepts": 3,
        "lexical_gap": "élevé",
    },
}


if not API_KEY:
    raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")
if not AZURE_ENDPOINT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")


client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)


# Ouvre la base Chroma enregistrée et récupère la collection de chunks
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return chroma_client.get_collection(COLLECTION_NAME)


# Renvoie chaque topic une seule fois, même s'il est associé à plusieurs chunks
def get_topics(collection):
    all_items = collection.get(include=["metadatas"], limit=50000)
    metadatas = all_items.get("metadatas", [])

    seen = set()
    topics = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        topic = meta.get("topic_name")
        if not topic or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
    return topics


# Charge les chunks d'un topic avec leur texte, leurs métadonnées et leurs vecteurs
def fetch_topic_chunks(collection, topic, limit=50000):
    items = collection.get(
        where={"topic_name": topic},
        include=["documents", "metadatas", "embeddings"],
        limit=limit,
    )

    chunks = []
    ids = items.get("ids", [])
    documents = items.get("documents", [])
    metadatas = items.get("metadatas", [])
    embeddings = items.get("embeddings", [])
    seen_chunk_ids = set()

    for index, document in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) else {}
        chunk_id = ids[index] if index < len(ids) else str(index)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        chunks.append(
            {
                "id": str(chunk_id),
                "text": document,
                "title": meta.get("title", "Unknown title"),
                "url": meta.get("url", ""),
                "topic_name": meta.get("topic_name", topic),
                "embedding": embeddings[index] if index < len(embeddings) else None,
            }
        )
    return chunks


def group_chunks_by_topic(chunks):
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["topic_name"]].append(chunk)
    return [sorted(group, key=lambda c: c["id"]) for group in grouped.values()]


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Pour chaque chunk non classé, regroupe avec lui les chunks dont la similarité atteint 0.8 (pour créer des sous-groupes au sein d'un topic)
def cluster_similar_chunks(chunks):
    if len(chunks) <= 1:
        return [chunks]

    clusters = []
    assigned = set()

    for i in range(len(chunks)):
        if i in assigned:
            continue
        current_cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(chunks)):
            if j in assigned:
                continue
            sim = cosine_similarity(chunks[i].get("embedding"), chunks[j].get("embedding"))
            if sim >= SIMILARITY_THRESHOLD:
                current_cluster.append(j)
                assigned.add(j)
        clusters.append([chunks[idx] for idx in current_cluster])

    if len(assigned) < len(chunks):
        for idx in range(len(chunks)):
            if idx not in assigned:
                clusters.append([chunks[idx]])

    return sorted(
        clusters,
        key=lambda group: (-len(group), min(int(chunk["id"]) for chunk in group if str(chunk["id"]).isdigit())),
    )


# Le prompt peut recevoir plusieurs chunks, mais la réponse doit en nécessiter seulement le nombre prévu ici 
def get_target_chunk_count(difficulty):
    profile = DIFFICULTY_PROFILES[difficulty]
    return profile["min_chunks"], profile["max_chunks"]


def get_candidate_chunk_count(difficulty):
    return DIFFICULTY_PROFILES[difficulty]["candidate_chunks"]


def classify_difficulty(cluster_size):
    if cluster_size <= 2:
        return "easy"
    if cluster_size <= 5:
        return "medium"
    return "hard"


def average_cluster_similarity(cluster):
    # Calcule la ressemblance moyenne entre toutes les paires de chunks du groupe.
    if len(cluster) <= 1:
        return 1.0

    similarities = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            similarities.append(cosine_similarity(cluster[i].get("embedding"), cluster[j].get("embedding")))

    if not similarities:
        return 1.0
    return float(sum(similarities) / len(similarities))


def rank_clusters_for_topic(topic_chunks, used_chunk_ids):
    candidate_clusters = cluster_similar_chunks(topic_chunks)
    ranked = []

    for group in candidate_clusters:
        valid_group = [chunk for chunk in group if chunk["id"] not in used_chunk_ids]
        if not valid_group:
            continue

        ranked.append(
            {
                "chunks": valid_group,
                "size": len(valid_group),
                "strength": average_cluster_similarity(valid_group),
            }
        )

    ranked.sort(key=lambda item: (-item["strength"], -item["size"]))
    return ranked


def select_topic_candidates(topic_chunks, difficulty, used_chunk_ids):
    # Choisit un groupe de la taille demandée, sans réutiliser les chunks d'une question précédente.
    target_size = get_candidate_chunk_count(difficulty)
    candidate_clusters = cluster_similar_chunks(topic_chunks)
    available_clusters = [
        [chunk for chunk in group if chunk["id"] not in used_chunk_ids]
        for group in candidate_clusters
    ]
    available_clusters = [group for group in available_clusters if len(group) >= target_size]

    if not available_clusters:
        return []

    exact_matches = [group for group in available_clusters if len(group) == target_size]
    if exact_matches:
        return max(exact_matches, key=average_cluster_similarity)

    return min(
        available_clusters,
        key=lambda group: (len(group) - target_size, -average_cluster_similarity(group)),
    )


# Prépare la consigne envoyée à GPT pour produire une question, sa réponse et le rôle de chaque chunk.
def build_question_prompt(topic, difficulty, chunks):
    profile = DIFFICULTY_PROFILES[difficulty]
    context_block = "\n\n---\n\n".join(
        (
            f"CHUNK_ID: {chunk['id']}\n"
            f"TITLE: {chunk['title']}\n"
            f"TOPIC_NAME: {chunk['topic_name']}\n"
            f"URL: {chunk['url']}\n"
            f"TEXT: {chunk['text']}"
        )
        for chunk in chunks
    )

    return f""" 
Tu es un générateur de benchmark de RAG.
Génère une question sur le topic \"{topic}\" de niveau de difficulté \"{difficulty}\".

La question doit être formulée de manière réaliste et doit nécessiter la combinaison de plusieurs informations présentes dans les chunks ci-dessous.

Contraintes explicites de difficulté :
- nombre de chunks nécessaires : entre {profile['min_chunks']} et {profile['max_chunks']}
- niveau de combinaison/synthèse : {profile['combination']}
- degré de reformulation : {profile['reformulation']}
- distance lexicale entre la question et les chunks : {profile['lexical_gap']}
- nombre de concepts à relier : {profile['concepts']}

Règles strictes :
- la question doit être répondable seulement à partir des informations présentes dans les chunks fournis
- la réponse ne doit pas être triviale ; elle doit nécessiter de relier plusieurs morceaux d'information
    - les chunks fournis sont uniquement des chunks candidats et ne sont pas tous nécessairement utiles
    - les chunks utiles doivent former un sous-groupe cohérent du topic {topic}
    - le nombre de chunks réellement nécessaires doit être compris entre {profile['min_chunks']} et {profile['max_chunks']}
- la question doit être claire, crédible, et non trop générique
- tous les chunks sélectionnés doivent être bien identifiables par leur CHUNK_ID et leur TITLE
- retourne strictement un JSON valide, sans texte autour, avec cette structure exacte :

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question": "...",
  "answer": "...",
  "necessary_chunk_ids": ["...", "..."],
  "necessary_titles": ["...", "..."],
  "redundant_chunk_ids": ["..."],
  "redundant_titles": ["..."],
  "irrelevant_chunk_ids": ["..."],
  "irrelevant_titles": ["..."],
  "difficulty_analysis": {{
      "min_chunks": {profile['min_chunks']},
      "max_chunks": {profile['max_chunks']},
      "combination": "{profile['combination']}",
      "reformulation": "{profile['reformulation']}",
      "lexical_gap": "{profile['lexical_gap']}",
      "concepts": {profile['concepts']}
  }}
}}

Voici les chunks disponibles :
{context_block}
"""


def normalize_ids(ids):
    seen = set()
    result = []
    for item in ids:
        cid = str(item)
        if cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def generate_question_for_topic(collection, topic, difficulty, used_chunk_ids, selected=None):
    # Si aucun groupe n'est fourni, récupère et sélectionne les chunks de ce topic
    if selected is None:
        topic_chunks = fetch_topic_chunks(collection, topic, limit=50000)
        if len(topic_chunks) < 2:
            raise ValueError(f"Pas assez de chunks pour produire une question sur le topic {topic}.")
        selected = select_topic_candidates(topic_chunks, difficulty, used_chunk_ids)

    if len(selected) < 2:
        raise ValueError(f"Trop peu de chunks cohérents pour le topic {topic} / {difficulty}.")

    # GPT reçoit les textes candidats et doit renvoyer les résultats au format JSON demandé
    prompt = build_question_prompt(topic, difficulty, selected)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Tu es un assistant qui génère des questions de benchmark RAG. Réponds uniquement en JSON valide."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )

    raw_text = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Réponse GPT invalide pour le topic {topic}: {raw_text[:500]}")
        data = json.loads(raw_text[start : end + 1])

    if not isinstance(data.get("necessary_chunk_ids"), list):
        raise ValueError(f"Réponse LLM invalide pour {topic} / {difficulty}: necessary_chunk_ids absent")

    if not isinstance(data.get("redundant_chunk_ids"), list):
        data["redundant_chunk_ids"] = []
    if not isinstance(data.get("redundant_titles"), list):
        data["redundant_titles"] = []
    if not isinstance(data.get("irrelevant_chunk_ids"), list):
        data["irrelevant_chunk_ids"] = []
    if not isinstance(data.get("irrelevant_titles"), list):
        data["irrelevant_titles"] = []

    # Vérifie que chaque chunk déclaré nécessaire faisait bien partie des textes envoyés à GPT
    data["necessary_chunk_ids"] = normalize_ids(data["necessary_chunk_ids"])
    selected_ids = {str(chunk["id"]) for chunk in selected}
    if not set(data["necessary_chunk_ids"]).issubset(selected_ids):
        raise ValueError(f"Le LLM a retourné un chunk nécessaire qui ne fait pas partie des candidats")

    min_necessary = DIFFICULTY_PROFILES[difficulty]["min_chunks"]
    max_necessary = DIFFICULTY_PROFILES[difficulty]["max_chunks"]
    num_necessary = len(data["necessary_chunk_ids"])
    if num_necessary < min_necessary or num_necessary > max_necessary:
        raise ValueError(
            f"Question {difficulty} rejetée: {num_necessary} chunk(s) nécessaire(s), "
            f"attendu entre {min_necessary} et {max_necessary}"
        )

    data["num_necessary_chunks"] = num_necessary
    data["necessary_titles"] = [
        next((chunk["title"] for chunk in selected if str(chunk["id"]) == cid), "Unknown title")
        for cid in data["necessary_chunk_ids"]
    ]

    for chunk in selected:
        used_chunk_ids.add(chunk["id"])

    return data


def generate_ground_truth():
    collection = get_collection()
    topics = get_topics(collection)
    if not topics:
        print("[GROUND_TRUTH] No topic found in the collection.")
        return []

    dataset = []
    used_chunk_ids = set()
    generated_count = 0

    print(f"[GROUND_TRUTH] Start: found {len(topics)} topic(s) in topic_name")

    # Pour chaque topic, tente de créer une question facile, moyenne puis difficile.
    for topic_index, topic in enumerate(topics, start=1):
        topic_chunks = fetch_topic_chunks(collection, topic, limit=50000)
        print(f"[GROUND_TRUTH] Topic {topic_index}/{len(topics)}: '{topic}'")

        for difficulty in ("easy", "medium", "hard"):
            target_size = get_candidate_chunk_count(difficulty)
            selected = select_topic_candidates(topic_chunks, difficulty, used_chunk_ids)
            if not selected:
                print(
                    f"[GROUND_TRUTH] SKIP -> topic='{topic}', difficulty={difficulty}: "
                    f"no cluster with at least {target_size} candidate chunks"
                )
                continue

            generated_count += 1
            print(
                f"[GROUND_TRUTH] [{generated_count}] topic='{topic}' | difficulty={difficulty} | "
                f"candidate_target={target_size} | candidate_chunks={len(selected)}"
            )
            accepted = False
            # Si GPT renvoie une sortie invalide ou qu'un contrôle échoue, la génération est retentée trois fois au maximum. Si toutes les tentatives échouent, la question est ignorée et le script passe à la suivante 
            for attempt in range(1, 4):
                try:
                    item = generate_question_for_topic(collection, topic, difficulty, used_chunk_ids, selected=selected)
                    item["topic"] = topic
                    item["difficulty"] = difficulty
                    dataset.append(item)
                    accepted = True
                    print(
                        f"[GROUND_TRUTH] OK -> question #{len(dataset)} saved for topic='{topic}' "
                        f"after attempt {attempt}"
                    )
                    break
                except Exception as exc:
                    print(
                        f"[GROUND_TRUTH] REJECT -> topic='{topic}', difficulty={difficulty}, "
                        f"attempt={attempt}/3: {exc}"
                    )

            if not accepted:
                print(f"[GROUND_TRUTH] SKIP -> no valid {difficulty} question for topic='{topic}'")

    print(f"[GROUND_TRUTH] Finished: {len(dataset)} questions generated from the strongest clusters only")
    return dataset


def save_ground_truth(dataset, output_path=OUTPUT_PATH):
    # Crée le dossier de destination si besoin et enregistre toutes les questions au format JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
    print(f"[GROUND_TRUTH] Saved {len(dataset)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark ground truth questions from Chroma chunks by topic and cluster.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Path to save the generated JSON.")
    args = parser.parse_args()

    dataset = generate_ground_truth()
    save_ground_truth(dataset, output_path=Path(args.output))


if __name__ == "__main__":
    main()


