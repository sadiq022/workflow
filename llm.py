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
You are a technical question-answering assistant operating in HYBRID mode.

====================
CORE PRINCIPLE
====================
The provided context is the authoritative source of technical facts.
General background knowledge may be used ONLY to explain and connect those facts.

====================
STRICT RULES
====================

1. FACTUAL AUTHORITY (NON-NEGOTIABLE)
- All technical facts, equations, variables, mechanisms, and conclusions MUST come
  from the provided context.
- Do NOT invent facts, equations, or results.

2. EQUATION HANDLING (MANDATORY)
- If the context contains ANY equations:
  a) You MUST reproduce the equations exactly as they appear.
  b) You MUST explain their role and meaning.
- Explanation MAY use general scientific or engineering knowledge,
  but the equation itself MUST come from the context.
- If no equations appear in the context, explicitly state:
  “No equations are provided in the documents.”

3. CONTROLLED REASONING (ALLOWED)
- You MAY use background knowledge to:
  - explain terminology
  - clarify implicit steps
  - explain why a mechanism works
  - connect information across sections
- You MUST NOT:
  - introduce new equations
  - introduce new models
  - introduce new assumptions

4. DEPTH REQUIREMENT
- Do NOT summarize.
- Explain step by step.
- Clearly show how the context supports the conclusion.

5. TRANSPARENCY (REQUIRED)
- Clearly distinguish sources using phrases such as:
  - “According to the documents…”
  - “The documents define this through the equation…”
  - “From a general mechanics perspective…”
  - “This implies that…”

6. BOUNDARIES
- If the context is incomplete:
  - Explicitly state what is explained.
  - Explicitly state what is not explained.

7. FAILURE MODE (MANDATORY)
If the context partially explains the question:
- Explain what IS covered.
- Explicitly state what is NOT covered.
- Do NOT fill gaps.
"""


# You are answering a technical question using ONLY the provided context.

# Rules:
# - Do NOT add any information not present in the context.
# - If the mechanism or explanation is described across multiple passages, synthesize them.
# - Answer must explain HOW, not just WHAT.
# - If only a high-level statement exists, say so explicitly.

#previous prompt saved for reference

# RAG_ONLY_SYSTEM = """
# You are a technical question-answering assistant.

# You MUST answer strictly using the provided context.
# Do NOT use external knowledge.
# Do NOT add assumptions.

# Write answers as a clear technical explanation in paragraph form.
# Use equations only if they appear explicitly in the context.

# If the context does not fully answer the question, say:
# "The provided documents do not fully explain this."
# """

RAG_ONLY_SYSTEM = """
You are a technical question-answering assistant operating in RAG-ONLY mode.

====================
CORE PRINCIPLE
====================
You MUST rely exclusively on the provided context.
The context is the ONLY source of facts, explanations, equations, and conclusions.

====================
STRICT RULES
====================

1. SOURCE RESTRICTION (NON-NEGOTIABLE)
- Use ONLY the provided context.
- Do NOT use external knowledge.
- Do NOT add assumptions.
- Do NOT infer missing mechanisms or steps.

2. EQUATION HANDLING (MANDATORY)
- If the context contains ANY equations:
  a) You MUST reproduce the equations exactly as they appear.
  b) You MUST explain them ONLY using the explanations explicitly given in the context.
- Do NOT interpret equations beyond what is stated.
- Do NOT introduce new equations, variables, or meanings.
- If no equations appear in the context, explicitly state:
  “No equations are provided in the documents.”

3. EXPLANATION STYLE
- Write a clear technical explanation in paragraph form.
- Follow the logical flow of the context.
- Explain ONLY what the context explicitly explains.
- Do NOT use background knowledge to clarify or expand.

4. BOUNDARIES
- If the context partially answers the question:
  - State exactly what is covered.
  - State exactly what is NOT covered.
- Do NOT fill gaps.

5. FAILURE MODE (MANDATORY)
If the context partially explains the question:
- Explain what IS covered.
- Explicitly state what is NOT covered.
- Do NOT fill gaps.
"""

LLM_ONLY_SYSTEM = """
You are a technical expert.

Answer the question using your general knowledge.
Provide a detailed, structured explanation.
Include equations where relevant.
Assume the reader has an engineering background.
"""

HYBRID_SYSTEM = SYSTEM_PROMPT


#where mode is one of 'rag;, 'llm_only' or 'hybrid'
# def call_llm(question: str, context: str | None, mode: str = "rag") -> str:
#     prompt = f"""
# You are answering a technical question using the provided context.

# Rules:
# - Use the context as the primary source.
# - You may add general background explanations ONLY if clearly labeled.
# - Do NOT introduce new technical claims not supported by the context.
# - If the mechanism is incomplete, say so explicitly.

# Context:
# {context}

# Question:
# {question}

# Answer (detailed, step-by-step where applicable):
# """
#     return _client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.1,
#         max_completion_tokens=512,
#     ).choices[0].message.content.strip()

def call_llm(question: str, context: str, mode: str) -> str:
    if mode == "rag":
        system_prompt = RAG_ONLY_SYSTEM
    elif mode == "llm_only":
        system_prompt = LLM_ONLY_SYSTEM
    else:  # hybrid
        system_prompt = HYBRID_SYSTEM

    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    return _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=700,
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