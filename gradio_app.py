import gradio as gr
from rag_search import rag_search
from llm import call_llm


def answer_question(query, mode):
    if not query.strip():
        return "Please enter a question.", ""

    if mode == "llm_only":
        answer = call_llm(question=query, context="", mode=mode)
        return answer, "LLM knowledge only"

    result = rag_search(query, mode)

    answer = result["answer"]
    references = "\n".join(
        f"{pdf}, page {page}"
        for pdf, page in result["references"]
    )

    return answer, references


with gr.Blocks(title="PDF RAG Assistant") as demo:
    gr.Markdown("## 📄 PDF RAG Assistant")
    gr.Markdown(
        "Test RAG, Hybrid, and LLM-only question answering."
    )

    # -----------------------
    # Inputs
    # -----------------------
    query = gr.Textbox(
        label="Your question",
        placeholder="How does multicontinuum theory extend to n-constituent composites?",
        lines=2,
    )

    mode = gr.Radio(
        choices=[
            ("Documents only", "rag"),
            ("LLM only (no documents)", "llm_only"),
            ("Hybrid (documents + LLM)", "hybrid"),
        ],
        label="Answering mode",
        value="rag",
    )

    ask_btn = gr.Button("Ask")

    # -----------------------
    # Outputs
    # -----------------------
    answer = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False,
    )

    references = gr.Textbox(
        label="References",
        lines=6,
        interactive=False,
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[query, mode],
        outputs=[answer, references],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5860,
        share=True,
    )
