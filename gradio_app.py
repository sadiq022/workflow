import gradio as gr
from rag_search import rag_search


def answer_question(query):
    if not query.strip():
        return "Please enter a question.", "", ""

    result = rag_search(query)

    answer = result["answer"]
    confidence = f"{result['confidence']:.2f}"

    references = "\n".join(
        f"{pdf}, page {page}"
        for pdf, page in result["references"]
    )

    return answer, confidence, references


with gr.Blocks(title="PDF RAG Assistant") as demo:
    gr.Markdown("## 📄 PDF RAG Assistant")
    gr.Markdown(
        "Ask questions grounded strictly in the uploaded research papers."
    )

    query = gr.Textbox(
        label="Your question",
        placeholder="How does multicontinuum theory extend to n-constituent composites?",
        lines=2,
    )

    ask_btn = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False,
    )

    confidence = gr.Textbox(
        label="Confidence",
        interactive=False,
    )

    references = gr.Textbox(
        label="References",
        lines=6,
        interactive=False,
    )

    ask_btn.click(
        fn=answer_question,
        inputs=query,
        outputs=[answer, confidence, references],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  
        server_port=5860,
        share=True,
    )
