import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\montaha.ben.zaied\Desktop\RAG\data")
BASE_FILE = DATA_DIR / "all_data_filtre_2006.csv" # fichier CSV contenant les lignes provenant des 52 fichiers csv séparés + filtrées à partir de 2006
TOPICS_FILE = DATA_DIR / "doc_topics.csv" # fichier CSV contenant les topics attribués aux white papers
OUTPUT_FILE = DATA_DIR / "data_finale.csv" # fichier CSV résultant de la fusion des deux bases

# Charger les deux bases
base = pd.read_csv(BASE_FILE)
doc_topics = pd.read_csv(TOPICS_FILE, encoding="cp1252")

# Nettoyer la clé de jointure avant la comparaison
for dataframe in (base, doc_topics):
    dataframe["url"] = dataframe["url"].astype("string").str.strip()

# doc_topics.csv contient les  4200 white papers auxquels Enrique a attribué un topic (ce qui réprésente une information supplémentaire très utile); on garde seulement ces white papers
urls_topics = doc_topics["url"].dropna().unique()
base_commune = base[base["url"].isin(urls_topics)].copy()

# Ajouter les colonnes correspondantes des topics pour ces URLs communes
data_finale = base_commune.merge(
    doc_topics.dropna(subset=["url"]),
    on="url",
    how="inner",
    suffixes=("_base", "_topics")
)

data_finale.to_csv(OUTPUT_FILE, index=False)

print(f"Lignes dans la base principale : {len(base)}")
print(f"Lignes dans doc_topics : {len(doc_topics)}")
print(f"Lignes de la base principale avec une URL commune : {len(base_commune)}")
print(f"Lignes dans data_finale : {len(data_finale)}")
print(f"Fichier créé : {OUTPUT_FILE}")
