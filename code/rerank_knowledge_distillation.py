import json
import os
import random
import ssl
from collections import defaultdict
from pathlib import Path

# Temporary workaround for enterprise proxy / MITM SSL interception.
# This disables certificate validation ONLY for this Python process.
# Use only in a controlled corporate environment where the cert is expected.
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    import requests
    requests.packages.urllib3.disable_warnings()
except Exception:
    pass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = ROOT / "data" / "fichiers json" / "GROUND_TRUTH_final.json"
CHUNKS_PATH = ROOT / "data" / "fichiers json" / "chunks.json"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_TO_SCORE = {
    "necessary": 2.0,
    "redundant": 1.0,
    "irrelevant": 0.0,
}

LABEL_PRIORITY = ["necessary", "redundant", "irrelevant"]
PAIR_ORDER = [
    ("necessary", "redundant"),
    ("necessary", "irrelevant"),
    ("redundant", "irrelevant"),
]


def load_chunks_by_id(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    mapping = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_id = metadata.get("chunk_id")
        text = chunk.get("text", "")
        if chunk_id is not None:
            mapping[str(chunk_id)] = text
    return mapping


def load_ground_truth(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("GROUND_TRUTH_final.json must contain a list of items.")
    return records


def build_query_examples(records, chunk_text_by_id):
    query_to_examples = defaultdict(list)

    for record in records:
        question = str(record.get("question", "")).strip()
        if not question:
            continue

        for label in LABEL_PRIORITY:
            for chunk_id in record.get(f"{label}_chunk_ids", []):
                text = chunk_text_by_id.get(str(chunk_id), "")
                if text:
                    query_to_examples[question].append(
                        {
                            "question": question,
                            "document": text,
                            "label": label,
                            "chunk_id": str(chunk_id),
                        }
                    )

    return query_to_examples


def build_query_pairs(query_to_examples):
    pairs = []

    for question, examples in query_to_examples.items():
        grouped = defaultdict(list)
        for example in examples:
            grouped[example["label"]].append(example["document"])

        for pos_label, neg_label in PAIR_ORDER:
            pos_docs = grouped.get(pos_label, [])
            neg_docs = grouped.get(neg_label, [])
            if not pos_docs or not neg_docs:
                continue
            for pos_doc in pos_docs:
                for neg_doc in neg_docs:
                    pairs.append(
                        {
                            "question": question,
                            "positive_doc": pos_doc,
                            "negative_doc": neg_doc,
                            "label": 1,
                        }
                    )

    return pairs


def split_by_query(pairs, val_fraction=0.2, seed=42):
    query_to_pairs = defaultdict(list)
    for pair in pairs:
        query_to_pairs[pair["question"]].append(pair)

    queries = list(query_to_pairs.keys())
    random.Random(seed).shuffle(queries)
    split_index = max(1, int(len(queries) * (1 - val_fraction)))

    train_queries = queries[:split_index]
    val_queries = queries[split_index:]

    train_pairs = []
    val_pairs = []
    for query, items in query_to_pairs.items():
        if query in train_queries:
            train_pairs.extend(items)
        else:
            val_pairs.extend(items)

    return train_pairs, val_pairs


def score_with_gpt4o(question: str, document: str):
    """
    Hook for a teacher LLM (GPT-4o). In production, replace this with your OpenAI client.
    For now, the ground-truth labels are used as the supervision signal.
    """
    return None


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return item["question"], item["positive_doc"], item["negative_doc"]


def collate_batch(batch):
    questions = [item[0] for item in batch]
    pos_docs = [item[1] for item in batch]
    neg_docs = [item[2] for item in batch]
    return questions, pos_docs, neg_docs


def train_student_reranker(train_pairs, val_pairs, model_name=MODEL_NAME, epochs=2, batch_size=8, lr=2e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(DEVICE)
    model.train()

    criterion = nn.MarginRankingLoss(margin=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        PairDataset(val_pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for questions, pos_docs, neg_docs in train_loader:
            pos_inputs = tokenizer(
                questions,
                pos_docs,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            neg_inputs = tokenizer(
                questions,
                neg_docs,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            pos_logits = model(**pos_inputs).logits.squeeze(-1)
            neg_logits = model(**neg_inputs).logits.squeeze(-1)
            targets = torch.ones_like(pos_logits)

            loss = criterion(pos_logits, neg_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        train_loss = total_loss / max(1, steps)
        val_loss = evaluate_pairwise_loss(model, tokenizer, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return model, tokenizer


def evaluate_pairwise_loss(model, tokenizer, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for questions, pos_docs, neg_docs in val_loader:
            pos_inputs = tokenizer(
                questions,
                pos_docs,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            neg_inputs = tokenizer(
                questions,
                neg_docs,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            pos_logits = model(**pos_inputs).logits.squeeze(-1)
            neg_logits = model(**neg_inputs).logits.squeeze(-1)
            targets = torch.ones_like(pos_logits)
            loss = criterion(pos_logits, neg_logits, targets)

            total_loss += loss.item()
            steps += 1

    return total_loss / max(1, steps)


def rank_candidates(model, tokenizer, question, documents, top_k=5):
    model.eval()
    scored = []

    for doc in documents:
        inputs = tokenizer(
            question,
            doc,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            score = model(**inputs).logits.squeeze(-1).item()

        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def main():
    records = load_ground_truth(GROUND_TRUTH_PATH)
    chunk_text_by_id = load_chunks_by_id(CHUNKS_PATH)
    query_to_examples = build_query_examples(records, chunk_text_by_id)
    pairs = build_query_pairs(query_to_examples)
    train_pairs, val_pairs = split_by_query(pairs, val_fraction=0.2, seed=42)

    print(f"Queries: {len(query_to_examples)}")
    print(f"Pairs total: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    model, tokenizer = train_student_reranker(train_pairs, val_pairs, epochs=2, batch_size=8)

    sample_question = next(iter(query_to_examples.keys()))
    sample_documents = [
        ex["document"] for ex in query_to_examples[sample_question][:5]
    ]

    ranked = rank_candidates(model, tokenizer, sample_question, sample_documents, top_k=3)
    print("\nExample ranking for a sampled question:")
    for doc, score in ranked:
        print(f"score={score:.4f} | {doc[:180]}...")


if __name__ == "__main__":
    main()
