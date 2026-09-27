import os
import re
from openai import OpenAI
from pydantic import BaseModel, Field


RERANK_MODEL = "gpt-4o"

client = OpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"], # La clé Azure est fournie comme variable d'environnement 
    base_url="https://openai-paris-rnd.openai.azure.com/openai/v1",
)


class RatingScore(BaseModel):
    # Refuse toute note qui ne se situe pas entre 1 et 10
    relevance_score: float = Field(
        ...,
        ge=1,
        le=10,
        description="Score de pertinence du document par rapport a la question.",
    )


def parse_relevance_score(content):
    # Extrait la note de la réponse, qu'elle soit précédée d'un libellé ou non
    score_match = re.search(
        r"(?:score|note|relevance_score)\D*(10(?:[.,]\d+)?|[1-9](?:[.,]\d+)?)",
        content.lower(),
    )
    if score_match is None:
        score_match = re.search(
            r"(?<!\d)(10(?:[.,]\d+)?|[1-9](?:[.,]\d+)?)(?!\d)",
            content,
        )
    if score_match is None:
        raise ValueError("Aucun score de pertinence trouve dans la reponse.")
    score = float(score_match.group(1).replace(",", "."))
    return RatingScore(relevance_score=score).relevance_score


def rerank_documents(query, documents, top_n=5):
    scored_documents = []
    for document in documents:
        # le LLM appelé évalue la pertinence de ce document par rapport à la question à partir du prompt qu'on introduit en input
        prompt = ( 
            "Evalue la pertinence du document par rapport a la question sur une "
            "echelle de 1 a 10. Tiens compte de l'intention et du contexte de la "
            "question, pas seulement des mots en commun. Reponds uniquement avec "
            "un nombre entre 1 et 10, sans explication.\n\n"
            f"Question : {query}\n\n"
            f"Document :\n{document['document']}"
        )
        try:
            response = client.chat.completions.create(
                model=RERANK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un evaluateur de pertinence documentaire.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=10,
            )
            content = response.choices[0].message.content or ""
            score = parse_relevance_score(content)
        except (AttributeError, IndexError, TypeError, ValueError):
            # Sans note valide, le score 0 place ce document après les résultats évalués
            score = 0.0

        scored_document = dict(document)
        scored_document["rerank_score"] = score
        scored_documents.append(scored_document)

    # Classe les documents du meilleur score au moins bon et garde les top_n premiers
    return sorted(
        scored_documents,
        key=lambda document: document["rerank_score"],
        reverse=True,
    )[:top_n]
