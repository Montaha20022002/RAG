import json
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "fichiers json" / "chunks.json"
EMBEDDINGS_PATH = ROOT / "data" / "fichiers json" / "embeddings.json"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "milliman_chunks"


def sanitize_metadata(metadata):
    # chromadb n'accepte que des valeurs simples dans les métadonnées
    clean = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        # Les valeurs NaN ne sont pas des métadonnées exploitables dans ChromaDB.
        if isinstance(value, float) and value != value:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def load_documents():
    # Les embeddings et les textes doivent être disponibles avant l'import
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CHUNKS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {EMBEDDINGS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("Le JSON chunks.json doit être une liste.")
    if not isinstance(embeddings, list):
        raise ValueError("Le JSON embeddings.json doit être une liste.")

    # Le texte sert de clé pour récupérer les métadonnées originales du chunk
    chunks_by_text = {}
    for item in chunks:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks_by_text[text] = item.get("metadata", {})

    documents = []
    embedding_vectors = []
    metadatas = []
    ids = []

    for idx, item in enumerate(embeddings):
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        emb = item.get("embedding")

        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(emb, list) or len(emb) == 0:
            continue

        metadata = item.get("metadata", {})
        source_metadata = chunks_by_text.get(text)
        if isinstance(source_metadata, dict) and source_metadata:
            metadata = source_metadata
        elif idx < len(chunks):
            candidate = chunks[idx]
            if isinstance(candidate, dict):
                metadata = candidate.get("metadata", metadata)

        # Reprend les métadonnées du chunk source pour conserver notamment topic_name.
        if isinstance(metadata, dict):
            chunk_match = None
            for candidate in chunks:
                if isinstance(candidate, dict) and candidate.get("text") == text:
                    chunk_match = candidate
                    break
            if isinstance(chunk_match, dict):
                metadata = chunk_match.get("metadata", metadata)

        # Ces listes gardent le même ordre pour associer chaque chunk à son embedding, à ses métadonnées et à son identifiant lors de l'import
        documents.append(text)
        embedding_vectors.append(emb)
        metadatas.append(sanitize_metadata(metadata))
        ids.append(str(idx))

    return documents, embedding_vectors, metadatas, ids


def main():
    documents, embeddings, metadatas, ids = load_documents()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # L'import repart de zéro : cette opération supprime les données de la collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # La distance cosine compare la direction des vecteurs d'embedding.
        metadata={"hnsw:space": "cosine"},
    )

    # L'ajout par lots limite la taille de chaque requête envoyée à ChromaDB
    batch_size = 5000 #valeur choisie arbitrairement car elle équilibre la taille des requêtes et la performance dans plusieurs travaux
    total = len(documents)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"Import batch {start}-{end-1}/{total}")

    count = collection.count()
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Nombre d'éléments importés : {count}")
    print(f"Base Chroma : {CHROMA_PATH}")


if __name__ == "__main__":
    main()
