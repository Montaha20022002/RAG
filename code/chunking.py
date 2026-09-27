from pathlib import Path
import pandas as pd
import json
import re

# 1. Fonction de nettoyage

def clean_text(text):
    
    if not isinstance(text, str):
        return ""
    # Supprime toutes les occurrences de [Doc: ...] (car elles contiennent les métadonnées qui sont déjà présentes dans les autres colonnes)
    text = re.sub(r'\[Doc:.*?\]', '', text)

    return text.strip()



# Dossier contenant les CSV
DATA_DIR = Path(r"C:/Users/montaha.ben.zaied/Desktop/RAG/data")

#Paramètres
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

#Fonction de chunking

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if not isinstance(text, str) or not text.strip():
        return []
    
    # Pour le chunking classique, on uniformise les espaces et retours à la ligne.
    text = re.sub(r'\s+', ' ', text).strip()
    
    # On découpe d'abord aux limites des phrases pour préserver leur contexte.
    sentences = re.findall(r'[^.!?]+[.!?]?|[^.!?]+$', text)

    def finalize_chunk(part):
        part = part.strip()
        if not part:
            return ""
        if not re.search(r'[.!?]$', part):
            part += "."
        return part

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                chunks.append(finalize_chunk(current))
                current = ""

            # Une phrase trop longue est découpée mot par mot pour respecter la limite.
            words = sentence.split()
            temp = ""
            for word in words:
                candidate = word if not temp else temp + " " + word
                if len(candidate) <= chunk_size:
                    temp = candidate
                else:
                    if temp:
                        chunks.append(finalize_chunk(temp))
                    temp = word
            if temp:
                current = temp
            continue

        candidate = sentence if not current else current + " " + sentence

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            previous = finalize_chunk(current)
            if previous:
                chunks.append(previous)
            # Le chevauchement conserve un peu de contexte entre deux chunks successifs.
            overlap_text = previous[-overlap:].lstrip() if overlap else ""
            current = overlap_text + " " + sentence if overlap_text else sentence

    if current:
        chunks.append(finalize_chunk(current))

    return chunks

# chunking de la colone contenu de l'échantillon 

all_chunks = []

# Chaque ligne du CSV est traitée comme un document, puis découpée séparément
csv_file = DATA_DIR / "echantillon.csv"


print(f"Traitement de : {csv_file.name}")
df = pd.read_csv(csv_file)
total_documents = len(df)
print(f"Nombre total de documents : {total_documents}")

# Chaque ligne = un document indépendant
for document_number, (row_index, row) in enumerate(df.iterrows(), start=1):

        # Récupération du contenu du document

        content = row["content"]

        # Nettoyage AVANT le chunking

        content = clean_text(content)

        # Si le contenu est NaN ou vide après nettoyage, on passe au suivant
        if pd.isna(row["content"]) or not content:
            print(f"Document {document_number}/{total_documents} ignoré : contenu vide")
            continue

        # Chunking du document

        print(f"Document {document_number}/{total_documents} : chunking en cours...", flush=True)
        chunks = chunk_text(content)
        print(
            f"Document {document_number}/{total_documents} terminé : "
            f"{len(chunks)} chunk(s) généré(s)",
            flush=True,
        )

        # Création des chunks + métadonnées ( titre , discipline , auteurs , url )

        for chunk_id, chunk in enumerate(chunks):

            # On recopie les métadonnées utiles pour filtrer ou identifier le chunk ensuite.
            allowed_columns = ["title", "discipline_base", "authors", "url"]
            metadata = {}
            for column in allowed_columns:
                if column not in df.columns:
                    continue

                value = row[column]
                if pd.isna(value):
                    continue

                if column == "authors":
                    value = str(value).strip()
                    if not value:
                        continue

                metadata[column] = value

            topic_name = row.get("topic_name") if "topic_name" in df.columns else None
            if pd.notna(topic_name):
                metadata["topic"] = topic_name
                metadata["topic_name"] = topic_name

            if "date_base" in df.columns and not pd.isna(row["date_base"]):
                try:
                    # L'année est dérivée de la date source pour faciliter les filtres temporels.
                    parsed_date = pd.to_datetime(row["date_base"], errors="coerce")
                    if pd.notna(parsed_date):
                        metadata["year"] = parsed_date.year
                except Exception:
                    metadata["year"] = None

            # Ce numéro repart à zéro pour chaque document source.
            metadata["chunk_id"] = chunk_id

            # Ajout du chunk final
            all_chunks.append({
                "text": chunk,
                "metadata": metadata
            })


# 5. Sauvegarde

output_file = DATA_DIR / "fichiers json" / "chunks.json"

# Le JSON conserve le texte de chaque chunk avec ses métadonnées associées.
with open(output_file, "w", encoding="utf-8") as f:

    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

# 6. Résumé

print("\nTerminé !")
print(f"Nombre de chunks : {len(all_chunks)}")
print(f"Fichier créé : {output_file}")