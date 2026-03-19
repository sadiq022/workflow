import os

# gRPC fix
os.environ['GRPC_KEEPALIVE_TIME_MS'] = '30000'
os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '10000'
os.environ['GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = 'true'
os.environ['GRPC_HTTP2_MIN_TIME_BETWEEN_PINGS_MS'] = '30000'

import gradio as gr
from rag_search import rag_search
from llm import call_llm
from audio_processing.mom_generator import run_pipeline
from pdf_upload import process_uploaded_pdfs, get_upload_status


# -----------------------
# HANDLERS
# -----------------------
def handle_query(query, mode):
    if not query.strip():
        return "Please enter a question.", ""

    if mode == "llm_only":
        return call_llm(query, "", mode), "LLM only"

    result = rag_search(query, mode)

    answer = result["answer"]
    refs = "\n".join(f"{pdf}, page {p}" for pdf, p in result["references"])

    return answer, refs


def handle_audio(audio_file):
    if audio_file is None:
        return "Please upload audio."

    yield "⏳ Processing audio... please wait..."

    result = run_pipeline(audio_file)

    yield result


def handle_pdf(files):
    if not files:
        return "No files selected."

    paths = [f.name for f in files]
    success, msg = process_uploaded_pdfs(paths)
    status = get_upload_status()

    return f"{msg}\n\n{status}"


# -----------------------
# UI
# -----------------------
with gr.Blocks(
    title="East-4D AI System",
    theme=gr.themes.Default(primary_hue="orange")
) as demo:

    gr.Markdown("# East-4D AI Information System")

    with gr.Tabs():

        # -----------------------
        # TAB 1: CHAT (MAIN)
        # -----------------------
        with gr.Tab("💬 Ask Questions"):

            query = gr.Textbox(
                placeholder="Ask something...",
                label="Question",
                lines=2
            )

            mode = gr.Radio(
                ["rag", "llm_only", "hybrid"],
                label="Mode",
                value="rag"
            )

            ask_btn = gr.Button("Submit", variant="primary")

            answer = gr.Markdown(label="Answer")

            refs = gr.Textbox(
                label="References",
                lines=4
            )

            ask_btn.click(
                fn=handle_query,
                inputs=[query, mode],
                outputs=[answer, refs]
            )

        # -----------------------
        # TAB 2: AUDIO → MoM
        # -----------------------
        with gr.Tab("🎧 Audio → MoM"):

            audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Upload Audio"
            )

            audio_btn = gr.Button("Generate MoM", variant="primary")

            mom_output = gr.Markdown(label="Minutes of Meeting")

            audio_btn.click(
                fn=handle_audio,
                inputs=audio,
                outputs=mom_output
            )

        # -----------------------
        # TAB 3: PDF Upload
        # -----------------------
        with gr.Tab("📄 Upload PDFs"):

            pdfs = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDFs"
            )

            upload_btn = gr.Button("Upload", variant="primary")

            upload_status = gr.Textbox(
                label="Status",
                lines=5
            )

            upload_btn.click(
                fn=handle_pdf,
                inputs=pdfs,
                outputs=upload_status
            )


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5860,
        share=True
    )


exit()

import os

# gRPC fix (Milvus stability)
os.environ['GRPC_KEEPALIVE_TIME_MS'] = '30000'
os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '10000'
os.environ['GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = 'true'
os.environ['GRPC_HTTP2_MIN_TIME_BETWEEN_PINGS_MS'] = '30000'

import gradio as gr
from rag_search import rag_search
from llm import call_llm
from audio_processing.mom_generator import run_pipeline


def unified_handler(query, audio_file, mode):
    """
    Single handler:
    - If audio → generate MoM
    - Else → normal RAG / LLM
    """

    # -----------------------
    # AUDIO MODE
    # -----------------------
    if audio_file is not None:
        try:
            mom_text = run_pipeline(audio_file)
            return mom_text, "Generated from audio"

        except Exception as e:
            return f"Error processing audio: {str(e)}", ""

    # -----------------------
    # TEXT MODE
    # -----------------------
    if not query.strip():
        return "Please enter a question or upload audio.", ""

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


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="East-4D AI Information System") as demo:

    gr.Markdown("## East-4D AI Information System")
    gr.Markdown(
        "Ask questions from documents OR upload audio to generate Minutes of Meeting."
    )

    # -----------------------
    # INPUTS (Single Interface)
    # -----------------------
    query = gr.Textbox(
        label="Enter your question (or leave empty if uploading audio)",
        placeholder="Ask something or upload audio below...",
        lines=2,
    )

    audio_input = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Upload Audio (optional)",
    )

    mode = gr.Radio(
        choices=[
            ("Documents only", "rag"),
            ("LLM only", "llm_only"),
            ("Hybrid (documents + LLM)", "hybrid"),
        ],
        label="Mode (for text queries)",
        value="rag",
    )

    ask_btn = gr.Button("Submit")

    # -----------------------
    # OUTPUTS
    # -----------------------
    answer = gr.Textbox(
        label="Answer / MoM",
        lines=15,
        interactive=False,
    )

    references = gr.Textbox(
        label="References / Info",
        lines=6,
        interactive=False,
    )

    # -----------------------
    # ACTION
    # -----------------------
    ask_btn.click(
        fn=unified_handler,
        inputs=[query, audio_input, mode],
        outputs=[answer, references],
        show_progress=True,
    )


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=5860,
        share=True,
    )
