"""
Transcribe audio files using WhisperX and save transcript
"""

import whisperx
import torch
import json
import os
from pathlib import Path
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("\nLoading WhisperX model (base)...")
model = whisperx.load_model(
    "base",
    device=device,
    compute_type="float16" if device == "cuda" else "float32"
)

print("Model loaded successfully")

def transcribe_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\nAudio file: {audio_path}")
    print(f"File size: {os.path.getsize(audio_path)/(1024*1024):.2f} MB")

    print("\nLoading audio...")
    audio = whisperx.load_audio(audio_path)

    print("\nTranscribing audio...")

    result = model.transcribe(
        audio,
        language="de",
        batch_size=8,
    )

    print("Transcription complete")

    # print("\nAligning transcript...")

    # model_a, metadata = whisperx.load_align_model(
    #     language_code=result["language"],
    #     device=device
    # )

    # result = whisperx.align(
    #     result["segments"],
    #     model_a,
    #     metadata,
    #     audio,
    #     device=device
    # )

    # print("Alignment complete")

    # print("\nPerforming speaker diarization...")

    # try:

    #     # hf_token = os.getenv("HF_TOKEN")

    #     diarize_model = DiarizationPipeline(
    #         device=device
    #     )

    #     diarize_segments = diarize_model(audio)

    #     result = whisperx.assign_word_speakers(
    #         diarize_segments,
    #         result
    #     )

    #     print("Diarization complete")

    # except Exception as e:

    #     print(f"Diarization skipped: {e}")

    return result


def save_transcript(result, audio_path):

    filename = Path(audio_path).stem

    json_file = os.path.join(OUTPUT_DIR, f"{filename}_transcript.json")
    txt_file = os.path.join(OUTPUT_DIR, f"{filename}_transcript.txt")

    full_text = "\n".join(
        f"{seg.get('speaker','Speaker')}: {seg['text']}"
        for seg in result["segments"]
    )

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved: {txt_file}")

    return txt_file