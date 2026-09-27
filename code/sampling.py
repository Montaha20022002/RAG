import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\montaha.ben.zaied\Desktop\RAG\data")
INPUT_FILE = DATA_DIR / "data_finale.csv"
SAMPLE_FILE = DATA_DIR / "echantillon_stratifie_discipline_topic.csv"
DISCIPLINE_COLUMN = "discipline_base"
TOPIC_COLUMN = "topic_name"
SAMPLE_FRACTION = 0.4
RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

# Vérifier que les colonnes utilisées pour le découpage existent

if DISCIPLINE_COLUMN not in df.columns or TOPIC_COLUMN not in df.columns:
    raise ValueError(
        f"Les colonnes '{DISCIPLINE_COLUMN}' et '{TOPIC_COLUMN}' sont nécessaires."
    )

# Chaque combinaison discipline-topic forme une strate, afin de préserver leur représentation dans l'échantillon
strata = df.groupby(
    [DISCIPLINE_COLUMN, TOPIC_COLUMN],
    dropna=False
)
sampled_strata = [
    group.sample(
     
        n=max(1, round(len(group) * SAMPLE_FRACTION)), # On garde au moins une ligne par strate, même pour les petits groupes.
        random_state=RANDOM_STATE
    )
    for _, group in strata
]
# Rassemble les groupes échantillonnés et conserve l'ordre initial des lignes
df_sample = (
    pd.concat(sampled_strata)
    .sort_index()
)

df_sample.to_csv(SAMPLE_FILE, index=False)

print(f"Nombre de lignes dans data_finale : {len(df)}")
print(f"Nombre de disciplines : {df[DISCIPLINE_COLUMN].nunique(dropna=False)}")
print(f"Nombre de topics : {df[TOPIC_COLUMN].nunique(dropna=False)}")
print(f"Nombre de strates discipline-topic : {len(strata)}")
print(f"Taille de l'échantillon : {len(df_sample)} lignes ({SAMPLE_FRACTION:.0%})")
print(f"Échantillon stratifié créé : {SAMPLE_FILE}")
