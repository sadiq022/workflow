import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# SYSTEM_PROMPT = """You are a question-answering assistant.

# You MUST answer strictly and only using the provided context.
# Do NOT use any external knowledge.
# If the answer is not contained in the context, say:
# "I cannot find the answer in the provided documents."

# Base your answer only on the context.
# """

SYSTEM_PROMPT = f"""
You are a technical question-answering assistant.

You must follow these rules strictly:

1. PRIMARY SOURCE:
   - Use the provided context as the primary source of truth.
   - Any factual claim that is present in the context MUST be based on it.

2. CONTROLLED BACKGROUND KNOWLEDGE:
   - You MAY use general background knowledge only to:
     - clarify terminology
     - explain standard concepts
     - improve readability
   - You MUST NOT introduce new technical claims, equations, or conclusions
     that are not present in the context.

3. TRANSPARENCY:
   - Clearly distinguish between:
     - information grounded in the provided documents
     - general background explanation
   - If the documents do not fully answer the question, say so explicitly.

4. CONSTRAINT:
   - If the answer depends on information NOT present in the context,
     do NOT guess or fabricate it.

5. FAILURE MODE:
   - If the context does not contain enough information to answer,
     say: "The provided documents do not fully explain this."
"""

# You are answering a technical question using ONLY the provided context.

# Rules:
# - Do NOT add any information not present in the context.
# - If the mechanism or explanation is described across multiple passages, synthesize them.
# - Answer must explain HOW, not just WHAT.
# - If only a high-level statement exists, say so explicitly.

#previous prompt saved for reference


def call_llm(question: str, context: str) -> str:
    prompt = f"""
You are answering a technical question using the provided context.

Rules:
- Use the context as the primary source.
- You may add general background explanations ONLY if clearly labeled.
- Do NOT introduce new technical claims not supported by the context.
- If the mechanism is incomplete, say so explicitly.

Context:
{context}

Question:
{question}

Answer (detailed, step-by-step where applicable):
"""
    return _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_completion_tokens=512,
    ).choices[0].message.content.strip()


def generate_search_queries(user_query: str) -> list[str]:
    """
    Generates alternative search queries for multi-query retrieval.
    This is NOT used for answering.
    """
    prompt = f"""
Generate 4 alternative search queries that could retrieve
relevant technical passages for the following question.

Rules:
- Do NOT answer the question
- Use different wording and terminology
- Focus on mechanisms, formulations, assumptions
- One query per line

Question:
{user_query}
"""

    resp = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=256,
    )

    lines = resp.choices[0].message.content.splitlines()
    return [l.strip("-• ").strip() for l in lines if l.strip()]