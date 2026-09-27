# Milliman RAG

Ce projet prépare un corpus de documents Milliman, le recherche avec un pipeline hybride, puis évalue les résultats à l'aide d'un ground truth.

## dossiers

- `code/` : scripts Python, organisés selon les étapes de préparation, d'indexation, de recherche et d'évaluation.
- `data/` : CSV de documents et de topics, échantillons, résultats d'analyse et fichiers JSON.
- `data/fichiers json/` : chunks, embeddings, ground truth et résultats des métriques.
- `chroma_db/` : base ChromaDB locale qui contient les chunks indexés et sert au retrieval et à la génération du ground truth.

## scripts

- `creation_data_finale.py` : associe les documents aux topics via leur URL et produit `data_finale.csv` à partir de `all_data_filtre_2006.csv`.
- `sampling.py` : crée un échantillon stratifié par discipline et topic.
- `chunking.py` : nettoie le contenu, le découpe en chunks et ajoute les métadonnées.
- `embeddings.py` : calcule un embedding pour chaque chunk.
- `chroma_import.py` : importe les chunks, embeddings et métadonnées dans ChromaDB.
- `ground_truth.py` : génère des questions de benchmark et leurs labels à partir des chunks indexés. Voir [GROUND_TRUTH_README.md](GROUND_TRUTH_README.md).
- `retrieval.py` : combine recherche vectorielle et BM25 avec RRF, puis appelle `rerank_LLM.py` pour classer les candidats avec GPT-4o.
- `test_retrieval.py` : évalue le retrieval avec le ground truth et enregistre les métriques et les latences.
- `topic_distribution.py`, `stat_des.py`, `duplication.py` : scripts d'analyse du corpus.
- `rerank_knowledge_distillation.py` : expérimentation distincte de fine-tuning d'un cross-encoder.

## le workflow

1. Regrouper et filtrer les documents, puis les associer aux topics.
2. Échantillonner les documents.
3. Découper leur contenu en chunks et conserver les métadonnées.
4. Calculer les embeddings et importer les chunks dans ChromaDB.
5. Générer ou charger le ground truth.
6. Lancer le retrieval hybride, le reranking LLM et l'évaluation.

## Régénérer `data/` et `chroma_db/`

Ces dossiers contiennent des fichiers très volumineux, et ne sont pas versionnés (voir `.gitignore`). Pour les reconstruire après un clone, exécuter dans l'ordre depuis `code/` : `creation_data_finale.py`, `sampling.py`, `chunking.py`, `embeddings.py`, puis `chroma_import.py`. Cela nécessite les CSV sources et un accès à l'API d'embeddings.

