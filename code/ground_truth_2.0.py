import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from openai import AzureOpenAI


ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "milliman_chunks"
OUTPUT_PATH = ROOT / "data" / "fichiers json" / "GROUND_TRUTH_2.0.json"
CHECKPOINT_PATH = ROOT / "data" / "fichiers json" / "GROUND_TRUTH_2.0_checkpoint.jsonl"

TEACHER_MODEL = os.getenv("AZURE_OPENAI_GROUND_TRUTH_MODEL", "gpt-5.4")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

QUESTIONS_PER_TOPIC_DIFFICULTY = 2
OVERSAMPLE_FACTOR = 1
MAX_TOP_UP_ROUNDS = 2
MIN_CHUNK_WORDS = 25
MAX_GROUP_ATTEMPTS_PER_QUESTION = 30
MAX_QUESTION_TOKEN_OVERLAP_HARD = 0.65
MAX_QUESTION_JACCARD_DUPLICATE = 0.88

STOPWORDS = {
	"a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
	"elle", "en", "et", "for", "from", "il", "ils", "in", "la", "le",
	"les", "leur", "mais", "of", "on", "or", "par", "pas", "pour", "que",
	"quel", "quelle", "quelles", "quels", "qui", "sont", "sur", "the", "un",
	"une", "what", "which", "with", "dans", "plus", "moins", "selon",
}

DIFFICULTY_PROFILES = {
	"easy": {
		"candidate_chunks": 2,
		"min_unique": 1,
		"max_unique": 2,
		"concept_count": 1,
		"reasoning_effort": "low",
	},
	"medium": {
		"candidate_chunks": 4,
		"min_unique": 2,
		"max_unique": 3,
		"concept_count": 2,
		"reasoning_effort": "medium",
	},
	"hard": {
		"candidate_chunks": 6,
		"min_unique": 3,
		"max_unique": 5,
		"concept_count": 3,
		"reasoning_effort": "medium",
	},
}
DIFFICULTIES = tuple(DIFFICULTY_PROFILES)
VALID_RELEVANCE = {"relevant", "irrelevant"}
VALID_CONTRIBUTIONS = {"unique", "overlapping"}


def clean_chunk_text(text):
	"""Normalize text for screening while retaining source wording for evidence checks."""
	if not isinstance(text, str):
		return ""
	text = re.sub(r"\[Doc:.*?\]", " ", text, flags=re.DOTALL)
	return re.sub(r"\s+", " ", text).strip()


def content_tokens(text):
	return {
		token.lower()
		for token in re.findall(r"[\w'-]+", text, flags=re.UNICODE)
		if len(token) > 2 and token.lower() not in STOPWORDS
	}


def normalize_for_match(text):
	return re.sub(r"\s+", " ", text).strip().casefold()


def quote_is_verbatim(quote, text):
	return bool(quote) and normalize_for_match(quote) in normalize_for_match(text)


def cosine_similarity(vector_a, vector_b):
	if vector_a is None or vector_b is None:
		return 0.0
	vector_a = np.asarray(vector_a, dtype=np.float32)
	vector_b = np.asarray(vector_b, dtype=np.float32)
	denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
	if denominator == 0:
		return 0.0
	return float(np.dot(vector_a, vector_b) / denominator)


def get_client():
	api_key = os.getenv("AZURE_OPENAI_API_KEY")
	endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
	if not api_key or not endpoint:
		raise RuntimeError(
			"AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT doivent être définis."
		)
	return AzureOpenAI(
		api_key=api_key,
		azure_endpoint=endpoint,
		api_version=API_VERSION,
	)


def call_json_model(client, model, instructions, prompt, effort, max_output_tokens=4000):
	response = client.responses.create(
		model=model,
		instructions=instructions,
		input=prompt,
		reasoning={"effort": effort},
		text={"format": {"type": "json_object"}},
		max_output_tokens=max_output_tokens,
	)
	status = getattr(response, "status", None)
	if status and status != "completed":
		details = getattr(response, "incomplete_details", None)
		reason = getattr(details, "reason", None)
		raise ValueError(
			f"Réponse incomplète de {model} (statut={status}, raison={reason})."
		)
	content = response.output_text
	if not content:
		raise ValueError(f"{model} n'a renvoyé aucun texte.")
	try:
		result = json.loads(content)
	except json.JSONDecodeError as exc:
		raise ValueError(
			f"JSON invalide de {model} (position {exc.pos}/{len(content)}: {exc.msg})."
		) from exc
	if not isinstance(result, dict):
		raise ValueError(f"La réponse de {model} doit être un objet JSON.")
	return result


def load_topic_chunks(collection, limit=50000):
	"""Fetch Chroma IDs (the IDs used by retrieval), text, metadata and vectors by topic."""
	topics = defaultdict(list)
	offset = 0
	page_size = 1000
	while True:
		page = collection.get(
			include=["metadatas"],
			limit=page_size,
			offset=offset,
		)
		ids = page.get("ids", [])
		metadatas = page.get("metadatas", []) or []
		for chunk_id, metadata in zip(ids, metadatas):
			if isinstance(metadata, dict) and metadata.get("topic_name"):
				topics[str(metadata["topic_name"])].append(str(chunk_id))
		if len(ids) < page_size:
			break
		offset += page_size

	result = {}
	for topic in topics:
		items = collection.get(
			where={"topic_name": topic},
			include=["documents", "metadatas", "embeddings"],
			limit=limit,
		)
		chunks = []
		seen_texts = set()
		for index, chunk_id in enumerate(items.get("ids", [])):
			text = clean_chunk_text(items["documents"][index])
			metadata = (items.get("metadatas") or [{}])[index] or {}
			words = content_tokens(text)
			if len(words) < MIN_CHUNK_WORDS:
				continue
			fingerprint = " ".join(sorted(words))
			if fingerprint in seen_texts:
				continue
			seen_texts.add(fingerprint)
			embedding = None
			if items.get("embeddings") is not None:
				embedding = items["embeddings"][index]
			chunks.append(
				{
					"id": str(chunk_id),
					"text": text,
					"metadata": metadata,
					"embedding": embedding,
				}
			)
		if chunks:
			result[topic] = chunks
	return result


def make_candidate_group(chunks, difficulty, rng):
	"""Choose a distinct same-topic group; hard groups span less-similar neighbors."""
	profile = DIFFICULTY_PROFILES[difficulty]
	group_size = profile["candidate_chunks"]
	if len(chunks) < group_size or any(chunk["embedding"] is None for chunk in chunks):
		return None

	seed = rng.choice(chunks)
	neighbors = [
		(cosine_similarity(seed["embedding"], chunk["embedding"]), chunk)
		for chunk in chunks
		if chunk["id"] != seed["id"]
	]
	neighbors.sort(key=lambda item: item[0], reverse=True)
	needed = group_size - 1
	if len(neighbors) < needed:
		return None

	if difficulty == "easy":
		selected = [item[1] for item in neighbors[:needed]]
	else:
		fractions = (
			[0.20, 0.55, 0.90]
			if difficulty == "medium"
			else [0.05, 0.25, 0.50, 0.75, 0.95]
		)
		indices = sorted(
			{min(len(neighbors) - 1, round(fraction * (len(neighbors) - 1))) for fraction in fractions}
		)
		selected = [neighbors[index][1] for index in indices]
		if len(selected) < needed:
			selected.extend(
				item[1]
				for item in neighbors
				if item[1]["id"] not in {chunk["id"] for chunk in selected}
			)
		selected = selected[:needed]

	group = [seed, *selected]
	return group if len({chunk["id"] for chunk in group}) == group_size else None


def question_signature(question):
	return " ".join(sorted(content_tokens(question)))


def is_duplicate_question(question, seen_questions):
	tokens = content_tokens(question)
	if not tokens:
		return True
	for previous in seen_questions:
		previous_tokens = content_tokens(previous)
		union = tokens | previous_tokens
		if union and len(tokens & previous_tokens) / len(union) >= MAX_QUESTION_JACCARD_DUPLICATE:
			return True
	return False


def passes_mechanical_checks(item, group, difficulty, seen_questions):
	profile = DIFFICULTY_PROFILES[difficulty]
	question = str(item.get("question", "")).strip()
	answer = str(item.get("answer", "")).strip()
	concepts = item.get("concepts")
	classifications = item.get("classifications")
	chunk_by_id = {chunk["id"]: chunk for chunk in group}

	if not question or not answer or not isinstance(concepts, list):
		return False, "question, answer ou concepts manquant"
	if len(concepts) != profile["concept_count"]:
		return False, "nombre de concepts incorrect"
	if not isinstance(classifications, list):
		return False, "classifications des chunks manquantes"

	classification_by_id = {}
	for classification in classifications:
		if not isinstance(classification, dict):
			return False, "format de classification invalide"
		chunk_id = str(classification.get("chunk_id", ""))
		relevance = classification.get("relevance")
		contribution = classification.get("contribution_type")
		if chunk_id not in chunk_by_id or relevance not in VALID_RELEVANCE:
			return False, "ID de chunk ou pertinence inconnue"
		if chunk_id in classification_by_id:
			return False, "chunk classé plusieurs fois"
		evidence = classification.get("evidence", "")
		if relevance == "relevant":
			if contribution not in VALID_CONTRIBUTIONS:
				return False, f"type de contribution invalide pour {chunk_id}"
			if not quote_is_verbatim(evidence, chunk_by_id[chunk_id]["text"]):
				return False, f"preuve de classification absente pour {chunk_id}"
		elif contribution is not None or evidence:
			return False, f"un chunk irrelevant ne doit pas avoir de contribution ni de preuve: {chunk_id}"
		classification_by_id[chunk_id] = {
			**classification,
			"chunk_id": chunk_id,
			"relevance": relevance,
			"contribution_type": contribution,
		}

	if set(classification_by_id) != set(chunk_by_id):
		return False, "tous les chunks candidats ne sont pas classés"
	unique_count = sum(
		classification["relevance"] == "relevant"
		and classification["contribution_type"] == "unique"
		for classification in classification_by_id.values()
	)
	if not profile["min_unique"] <= unique_count <= profile["max_unique"]:
		return False, "nombre de contributions uniques incompatible avec le niveau"

	if is_duplicate_question(question, seen_questions):
		return False, "question vide ou trop proche d'une question déjà générée"

	if difficulty == "hard":
		question_tokens = content_tokens(question)
		source_tokens = set().union(*(content_tokens(chunk["text"]) for chunk in group))
		overlap = len(question_tokens & source_tokens) / max(1, len(question_tokens))
		if overlap > MAX_QUESTION_TOKEN_OVERLAP_HARD:
			return False, f"recouvrement lexical trop élevé pour hard ({overlap:.2f})"

	item["evidence"] = [
		{
			"chunk_id": chunk_id,
			"quote": classification["evidence"],
			"contribution_type": classification["contribution_type"],
		}
		for chunk_id, classification in classification_by_id.items()
		if classification["relevance"] == "relevant"
	]

	return True, "ok"


def build_generation_prompt(topic, difficulty, group):
	profile = DIFFICULTY_PROFILES[difficulty]
	chunks_block = "\n\n".join(
		f"CHUNK_ID: {chunk['id']}\nTITLE: {chunk['metadata'].get('title', '')}\nTEXT:\n{chunk['text']}"
		for chunk in group
	)
	return f"""Create one RAG benchmark example about topic {topic!r} at the {difficulty!r} difficulty level.
Write all generated natural-language content in English: concepts, answer, and question.
Use only the candidate chunks below. First identify {profile['concept_count']} concise concepts,
then provide a concise answer and question, and classify every candidate chunk.
The question must require reasoning appropriate to its difficulty; do not ask only for a title,
date, or isolated sentence. For hard questions, substantially paraphrase the source vocabulary.
Keep the output concise. For every relevant chunk, provide one short, exact quotation copied from
that chunk. Keep quotations verbatim even when the source chunk is not in English.

Use two separate classification fields:
- relevance: relevant or irrelevant.
- contribution_type: unique if the relevant information is not covered by another candidate;
  overlapping if another candidate already covers the same relevant information.
For an irrelevant chunk, contribution_type must be JSON null and evidence must be an empty string.
Each candidate chunk must appear exactly once. The number of unique relevant chunks must match
the difficulty constraints. Do not include explanations or extended difficulty analysis.

Return only a valid JSON object with exactly these fields:
{{"concepts":["short concept in English"],"answer":"concise answer in English",
"question":"question in English",
"classifications":[{{"chunk_id":"relevant_chunk_id","relevance":"relevant","contribution_type":"unique","evidence":"exact source quotation"}},
{{"chunk_id":"irrelevant_chunk_id","relevance":"irrelevant","contribution_type":null,"evidence":""}}]}}

Topic: {topic}
Difficulty: {difficulty}
Required number of concepts: {profile['concept_count']}
Candidate chunks:
{chunks_block}"""


def generate_candidates_for_target(
	client,
	topic,
	difficulty,
	chunks,
	count,
	rng,
	used_group_signatures,
	seen_questions,
):
	generated = []
	attempts = 0
	max_attempts = max(count * MAX_GROUP_ATTEMPTS_PER_QUESTION, 1)
	while len(generated) < count and attempts < max_attempts:
		attempts += 1
		group = make_candidate_group(chunks, difficulty, rng)
		if not group:
			break
		group_signature = (topic, difficulty, tuple(sorted(chunk["id"] for chunk in group)))
		if group_signature in used_group_signatures:
			continue
		used_group_signatures.add(group_signature)

		profile = DIFFICULTY_PROFILES[difficulty]
		try:
			item = call_json_model(
				client,
				TEACHER_MODEL,
				"Create RAG benchmark examples grounded strictly in the supplied source chunks. Write generated content in English.",
				build_generation_prompt(topic, difficulty, group),
				effort=profile["reasoning_effort"],
			)
		except ValueError as exc:
			print(f"REJECT generation topic={topic!r} difficulty={difficulty}: {exc}")
			continue
		valid, reason = passes_mechanical_checks(item, group, difficulty, seen_questions)
		if not valid:
			print(f"REJECT generation topic={topic!r} difficulty={difficulty}: {reason}")
			continue

		item.update(
			{
				"topic": topic,
				"difficulty": difficulty,
				"candidate_chunk_ids": [chunk["id"] for chunk in group],
				"_candidate_chunks": group,
			}
		)
		seen_questions.append(item["question"])
		generated.append(item)
		print(f"Generated candidate {len(generated)}/{count}: {topic} / {difficulty}")

	return generated
def finalize_candidate(candidate):
	"""Convert an already validated model response into the ground-truth record."""
	chunks = candidate["_candidate_chunks"]
	classifications = {
		str(item["chunk_id"]): item for item in candidate["classifications"]
	}
	relevant_ids = [
		chunk_id for chunk_id, item in classifications.items()
		if item["relevance"] == "relevant"
	]
	unique_ids = [
		chunk_id for chunk_id, item in classifications.items()
		if item["relevance"] == "relevant" and item["contribution_type"] == "unique"
	]
	overlapping_ids = [
		chunk_id for chunk_id, item in classifications.items()
		if item["relevance"] == "relevant" and item["contribution_type"] == "overlapping"
	]
	irrelevant_ids = [
		chunk_id for chunk_id, item in classifications.items()
		if item["relevance"] == "irrelevant"
	]
	result = {
		key: value for key, value in candidate.items()
		if not key.startswith("_") and key != "classifications"
	}
	result.update(
		{
			"relevant_chunk_ids": relevant_ids,
			"unique_chunk_ids": unique_ids,
			"overlapping_chunk_ids": overlapping_ids,
			"irrelevant_chunk_ids": irrelevant_ids,
			"candidate_titles": {
				chunk["id"]: chunk["metadata"].get("title", "") for chunk in chunks
			},
			"classification_evidence": classifications,
			"num_unique_chunks": len(unique_ids),
			"needs_human_review": True,
		}
	)
	return result


def build_quotas(topic_chunks, pilot=False):
	return {
		(topic, difficulty): (1 if pilot else QUESTIONS_PER_TOPIC_DIFFICULTY)
		for topic in topic_chunks
		for difficulty in DIFFICULTIES
	}


def generate_candidate_pool(client, topic_chunks, requested, rng, group_signatures, seen_questions):
	pool = []
	for (topic, difficulty), final_quota in requested.items():
		desired_candidates = final_quota * OVERSAMPLE_FACTOR
		generated = generate_candidates_for_target(
			client,
			topic,
			difficulty,
			topic_chunks[topic],
			desired_candidates,
			rng,
			group_signatures,
			seen_questions,
		)
		pool.extend(generated)
	return pool


def append_checkpoint(path, record):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "a", encoding="utf-8") as file:
		file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint(path, valid_topics):
	if not path.exists():
		return []

	records = []
	seen = set()
	with open(path, "r", encoding="utf-8") as file:
		for line_number, line in enumerate(file, start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError:
				print(f"Checkpoint ignoré ligne {line_number}: JSON invalide.")
				continue

			question = str(record.get("question", "")).strip()
			topic = record.get("topic")
			difficulty = record.get("difficulty")
			if not question or topic not in valid_topics or difficulty not in DIFFICULTIES:
				continue
			label_fields = (
				"relevant_chunk_ids",
				"unique_chunk_ids",
				"overlapping_chunk_ids",
				"irrelevant_chunk_ids",
			)
			if not all(isinstance(record.get(field), list) for field in label_fields):
				print(f"Checkpoint ignoré ligne {line_number}: ancien format de labels.")
				continue
			key = (topic, difficulty, question.casefold())
			if key in seen:
				continue
			seen.add(key)
			records.append(record)
	return records


def assess_candidates(candidates, checkpoint_path):
	accepted = []
	for index, candidate in enumerate(candidates, start=1):
		try:
			record = finalize_candidate(candidate)
			accepted.append(record)
			append_checkpoint(checkpoint_path, record)
			print(f"Accepted {index}/{len(candidates)}: {record['topic']} / {record['difficulty']}")
		except Exception as exc:
			print(
				f"REJECT validation topic={candidate['topic']!r} "
				f"difficulty={candidate['difficulty']}: {exc}"
			)
	return accepted


def select_final_questions(accepted, quotas):
	selected = []
	selected_counts = defaultdict(int)
	for record in accepted:
		key = (record["topic"], record["difficulty"])
		if selected_counts[key] >= quotas.get(key, 0):
			continue
		selected.append(record)
		selected_counts[key] += 1
	missing = {
		key: quota - selected_counts[key]
		for key, quota in quotas.items()
		if selected_counts[key] < quota
	}
	return selected, missing


def save_json(records, output_path):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as file:
		json.dump(records, file, ensure_ascii=False, indent=2)


def run_pipeline(output_path, checkpoint_path, seed=42, pilot_topics=0):
	chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
	collection = chroma.get_collection(COLLECTION_NAME)
	topic_chunks = load_topic_chunks(collection)
	if pilot_topics:
		selected_topics = sorted(topic_chunks)[:pilot_topics]
		topic_chunks = {topic: topic_chunks[topic] for topic in selected_topics}

	if not topic_chunks:
		raise ValueError("Aucun topic avec assez de chunks propres et d'embeddings dans ChromaDB.")

	quotas = build_quotas(topic_chunks, pilot=bool(pilot_topics))
	target_total = sum(quotas.values())
	accepted = load_checkpoint(checkpoint_path, set(topic_chunks))
	selected, missing = select_final_questions(accepted, quotas)
	requested = dict(missing)
	requested_candidates = sum(requested.values()) * OVERSAMPLE_FACTOR
	print(f"Topics retenus: {len(topic_chunks)}")
	print(f"Quota final demandé: {target_total} questions")
	print(f"Questions réutilisées depuis le checkpoint: {len(selected)}")
	print(f"Candidats à générer au premier passage: {requested_candidates}")
	if pilot_topics:
		print("Mode pilote actif : une question par niveau et par topic retenu.")

	client = get_client()
	rng = random.Random(seed)
	group_signatures = {
		(record["topic"], record["difficulty"], tuple(sorted(record.get("candidate_chunk_ids", []))))
		for record in accepted
		if record.get("candidate_chunk_ids")
	}
	seen_questions = [record["question"] for record in accepted]
	for round_index in range(MAX_TOP_UP_ROUNDS + 1):
		if not requested:
			break
		candidates = generate_candidate_pool(
			client,
			topic_chunks,
			requested,
			rng,
			group_signatures,
			seen_questions,
		)
		accepted.extend(assess_candidates(candidates, checkpoint_path))
		selected, missing = select_final_questions(accepted, quotas)
		if not missing:
			break
		if round_index == MAX_TOP_UP_ROUNDS:
			break
		requested = {key: count for key, count in missing.items()}
		print(f"Complément ciblé: {sum(requested.values())} places de quota restent à remplir.")

	selected, missing = select_final_questions(accepted, quotas)
	save_json(selected, output_path)
	print(f"Ground truth sauvegardé : {output_path}")
	print(f"Questions retenues : {len(selected)}/{target_total}")
	if missing:
		print("Quotas non atteints:")
		for (topic, difficulty), count in missing.items():
			print(f"  {topic} / {difficulty}: {count} manquante(s)")
	print(f"Candidats validés disponibles dans le checkpoint : {checkpoint_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Génère un ground truth RAG avec annotations et contrôles mécaniques."
	)
	parser.add_argument("--run", action="store_true", help="Confirme les appels Azure OpenAI potentiellement nombreux et coûteux.")
	parser.add_argument("--pilot-topics", type=int, default=0, help="Teste le pipeline sur N topics, avec une question par niveau.")
	parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
	parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	if not args.run:
		print("Aucun appel modèle lancé. Ajouter --run pour confirmer la génération.")
		print(f"Modèle de génération et d'annotation: {TEACHER_MODEL}")
		print(f"Sortie: {args.output}")
		print("Commencer par un pilote: python code/ground_truth_2.0.py --run --pilot-topics 1")
		return

	run_pipeline(
		output_path=args.output,
		checkpoint_path=args.checkpoint,
		seed=args.seed,
		pilot_topics=args.pilot_topics,
	)


if __name__ == "__main__":
	main()
