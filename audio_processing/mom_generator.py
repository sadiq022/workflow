import os
import json
from groq import Groq
from dotenv import load_dotenv
from audio_processing.transcribe import transcribe_audio, save_transcript

load_dotenv()

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# PASS 1: Extract structured meeting information
# ---------------------------------------------------

def extract_meeting_structure(transcript):

    prompt = f"""
You are analyzing a meeting transcript.

Extract structured meeting information.

Return ONLY valid JSON with this structure:

{{
 "meeting_title":"",
 "date":"",
 "leader":"",
 "participants":"",
 "goal":"",
 "topics":[
   {{
     "title":"",
     "key_points":[],
     "observations":[],
     "decisions":[],
     "actions":[]
   }}
 ]
}}

Rules:
- Preserve operational details.
- Do not summarize aggressively.
- Extract all major discussion topics.
- Topics must reflect actual discussion sections.

Transcript:
{transcript}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You extract structure from meeting transcripts."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------
# PASS 2: Generate detailed Minutes of Meeting
# ---------------------------------------------------

def generate_detailed_mom(structured_data):

    prompt = f"""
You are a professional meeting secretary.

Generate a detailed and well-structured Minutes of Meeting (MoM) from the provided structured meeting data.

Requirements:
- Include meeting title, date, leader, participants, and goal.
- Cover all discussion topics, decisions, observations, and actions.
- Preserve important details (do not over-summarize).
- Do not add information not present.

Structure:

## Meeting Details
- Title:
- Date:
- Participants:
- Goal:

## Discussion Topics
- Organize topics into numbered sections
- Clearly explain key points, observations, and discussions

## Decisions
- Present in a table

## Action Items
- Present in a table with columns: Action | Owner | Description

Formatting:
- Use Markdown
- Use tables for decisions and actions
- Keep professional and clear writing

Generate the MoM in the original meeting language.

Structured Data:
{structured_data}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You generate detailed professional meeting minutes."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def run_pipeline(audio_file):

    print("\n===== TRANSCRIBING AUDIO =====")

    result = transcribe_audio(audio_file)

    transcript_file = save_transcript(result, audio_file)

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    print("\n===== EXTRACTING MEETING STRUCTURE =====")

    structured_data = extract_meeting_structure(transcript)

    json_file = transcript_file.replace(".txt", "_structured.json")

    try:
        parsed = json.loads(structured_data)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print("Structured JSON saved")

    except Exception:
        print("Warning: JSON parsing failed. Saving raw output.")

        with open(json_file, "w", encoding="utf-8") as f:
            f.write(structured_data)

    print("\n===== GENERATING DETAILED MOM =====")

    mom_text = generate_detailed_mom(structured_data)

    mom_file = transcript_file.replace(".txt", "_MoM.txt")

    with open(mom_file, "w", encoding="utf-8") as f:
        f.write(mom_text)

    print("\nMoM saved:")
    print(mom_file)
    return mom_text


# ---------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------

if __name__ == "__main__":

    AUDIO_FILE = "/mnt/c/Users/AI PC/Downloads/34-Videos_Wassertropfen_Treffen/Wassertröpfchen 26.09.2025.m4a"

    run_pipeline(AUDIO_FILE)