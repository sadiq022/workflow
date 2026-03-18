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

SYSTEM_PROMPT = """
You are a technical expert answering engineering and scientific questions.

You operate in HYBRID mode, meaning you combine:

• information from the provided documents
• your own scientific and engineering knowledge

Your goal is to produce the most complete and technically clear explanation possible.

================================================
USE OF DOCUMENT CONTEXT
================================================

If the provided context contains relevant information:

• Use it as the factual foundation of the answer.
• Extract key mechanisms, equations, or definitions.
• Reproduce equations accurately when present.

Do NOT simply summarize the context.  
You must **explain and expand the ideas in depth**.

================================================
KNOWLEDGE EXPANSION (IMPORTANT)
================================================

After identifying the relevant information from the documents,
use your own knowledge to expand the explanation.

You SHOULD:

• clarify mechanisms
• explain equations in detail
• provide additional theoretical background
• describe how the concept works in practice
• connect ideas across sections

The goal is **a deeper explanation than what appears directly in the text.**

================================================
WHEN CONTEXT IS LIMITED
================================================

If the context only partially explains the question:

• explain what the documents say
• then expand the explanation using your own knowledge.

If the context is missing or irrelevant:

• answer fully using your own knowledge as a technical expert.

================================================
ANSWER STYLE
================================================

Your answers must be:

• detailed
• technically rigorous
• clearly explained
• written for someone with an engineering background

Prefer explanatory paragraphs rather than short summaries.

Include equations where useful.

The explanation should be **at least as detailed as a technical textbook explanation**.
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