# Construction du ground truth

Ce document décrit comment le benchmark de questions RAG est construit avec `code/ground_truth.py`.

## Données nécessaires

La génération utilise la collection ChromaDB `milliman_chunks`. Elle contient les textes des chunks, leurs embeddings et leurs métadonnées, notamment `topic_name`, `title` et `url`. Le ground truth est donc construit après la préparation des chunks, le calcul des embeddings et leur import dans ChromaDB.

Le script utilise Azure OpenAI. Variables requises : `AZURE_OPENAI_API_KEY` et `AZURE_OPENAI_ENDPOINT`. `AZURE_OPENAI_API_VERSION` et `AZURE_OPENAI_MODEL` . Le modèle configuré par défaut n'est pas GPT-4o, sauf si la variable d'environnement le désigne.

## Étapes de génération

1. **Lister les topics** — le script lit les métadonnées ChromaDB et récupère chaque `topic_name` une seule fois.
2. **Récupérer les chunks** — pour chaque topic, il charge les chunks, textes, métadonnées et embeddings correspondants.
3. **Former des groupes de candidats** — il compare les embeddings avec la similarité cosinus. À partir de chaque chunk non encore affecté, il ajoute au même groupe les chunks suivants dont la similarité atteint `0.8`. Il s'agit d'un regroupement glouton.
4. **Choisir les candidats selon la difficulté** — le script vise 2 chunks pour `easy`, 4 pour `medium` et 6 pour `hard`(chunks candidats). S'il n'existe pas de groupe de cette taille exacte, il peut choisir un groupe un peu plus grand. La réponse doit réellement nécessiter respectivement 1–2, 2–3 et 3–5 chunks.
5. **Demander une question au LLM appelé** — pour chaque topic, le script tente les niveaux facile, moyen et difficile (en respectant plusieurs critères portant sur le nombre des concepts, la difficulté lexicale etc). Le modèle reçoit les textes candidats et doit renvoyer une question, une réponse, les IDs/titres des chunks nécessaires, redondants et non pertinents (pour identifier les chunks d'une manière unique), ainsi qu'une analyse de difficulté.
6. **Valider la réponse** — le JSON doit contenir une liste de chunks nécessaires, ceux-ci doivent appartenir aux candidats envoyés, et leur nombre doit respecter la plage du niveau. Les champs de chunks redondants et non pertinents sont initialisés à des listes vides s'ils sont absents ou mal formés. En cas d'erreur de génération ou de validation, le script réessaie jusqu'à trois fois, puis ignore cette question.
7. **Éviter de réutiliser les candidats acceptés** — après une génération acceptée, les chunks du groupe sont marqués comme utilisés pour éviter de les reprendre pour les questions suivantes.
8. **Enregistrer le résultat** — le dataset est écrit en JSON avec la question, la réponse, le topic, la difficulté et les IDs/titres associés à chaque label.

## Labels du ground truth

- `necessary_chunk_ids` : chunks nécessaires pour répondre.
- `redundant_chunk_ids` : chunks liés au sujet, mais non nécessaires à la réponse.
- `irrelevant_chunk_ids` : chunks qui n'aident pas à répondre.

Ces labels fournissent un ordre de pertinence utilisable pour évaluer le retrieval ou entraîner un reranker.

## output
Le ground truth existant se trouve dans `data/fichiers json/GROUND_TRUTH_final.json`