import json
import os
from pathlib import Path
from openai import OpenAI

# L'appel de l'API du text embedding (fournie par Quincy)

ENDPOINT = "https://openai-paris-rnd.openai.azure.com/openai/v1"
MODEL = "text-embedding-3-large"
API_KEY = os.environ["AZURE_OPENAI_API_KEY"] # La clé API est récupérée depuis la variable d'environnement AZURE_OPENAI_API_KEY ( saisie dans le terminal)
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "fichiers json" / "chunks.json"
EMBEDDINGS_PATH = ROOT / "data" / "fichiers json" / "embeddings.json"

client = OpenAI(

    api_key=API_KEY,
    base_url=ENDPOINT.rstrip("/") + "/" 
)

# Charge les chunks déjà créés par le script de chunking (code/chunking.py)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Nombre de chunks : {len(chunks)}")

# Calculer un vecteur pour chaque texte et le conserve avec ses métadonnées.

for i, chunk in enumerate(chunks):

    text = chunk["text"]
    
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )

    chunk["embedding"] = response.data[0].embedding
    print(f"Embedding généré : {i + 1}/{len(chunks)}")

# Enregistrer les chunks et leurs vecteurs pour l'import ultérieur dans ChromaDB.

with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

print("Terminé !")
print(f"Fichier créé : {EMBEDDINGS_PATH}")
