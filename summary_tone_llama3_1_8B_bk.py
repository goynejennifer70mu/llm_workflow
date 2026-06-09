import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os
import os

os.environ["HF_TOKEN"] = "####"

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ---------------------------------------------------
# GPU CHECK
# ---------------------------------------------------
if torch.cuda.is_available():

    print("GPU detected")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

else:

    print("GPU not detected. Using CPU.")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
print("Loading Llama 3.1 8B model...")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded")

# ---------------------------------------------------
# NORMALIZE TONE
# ---------------------------------------------------
def normalize_tone(tone_text):

    tone_text = tone_text.lower()

    if "positive" in tone_text:
        return "Positive"

    elif "negative" in tone_text:
        return "Negative"

    else:
        return "Neutral"

# ---------------------------------------------------
# SPLIT LARGE TEXT
# ---------------------------------------------------
def chunk_text(text, max_words=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):

        chunks.append(
            " ".join(words[i:i + max_words])
        )

    return chunks

# ---------------------------------------------------
# GENERATE RESPONSE
# ---------------------------------------------------
def generate_response(prompt, max_new_tokens=200):

    full_prompt = f"""
You are an expert analyst for student course evaluations.

Provide concise, professional summaries.

{prompt}
"""

    output = pipe(
        full_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    return output[0]["generated_text"].strip()

# ---------------------------------------------------
# SUMMARIZE SINGLE CHUNK
# ---------------------------------------------------
def summarize_chunk(chunk):

    prompt = f"""
Summarize the following student feedback in 1-2 concise sentences.

Focus on:
- major themes
- instructor strengths
- concerns
- repeated feedback

Feedback:
{chunk}
"""

    return generate_response(
        prompt,
        max_new_tokens=150
    )

# ---------------------------------------------------
# MERGE SUMMARIES
# ---------------------------------------------------
def final_summary(chunk_summaries):

    combined = "\n".join(chunk_summaries)

    prompt = f"""
Combine the following summaries into a single professional summary.

Requirements:
- 2-4 sentences
- concise
- objective
- highlight major themes

Summaries:
{combined}
"""

    return generate_response(
        prompt,
        max_new_tokens=200
    )

# ---------------------------------------------------
# FULL SUMMARY PIPELINE
# ---------------------------------------------------
def summarize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    chunks = chunk_text(text)

    chunk_summaries = []

    for chunk in chunks:

        summary = summarize_chunk(chunk)

        chunk_summaries.append(summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    return final_summary(chunk_summaries)

# ---------------------------------------------------
# TONE CLASSIFICATION
# ---------------------------------------------------
def classify_tone(text):

    prompt = f"""
Analyze the overall tone of the following student feedback.

Return ONLY ONE WORD:
Positive
Negative
Neutral

Feedback:
{text}
"""

    result = generate_response(
        prompt,
        max_new_tokens=10
    )

    return normalize_tone(result)

# ---------------------------------------------------
# PROCESS CSV
# ---------------------------------------------------
def process_csv(
    input_csv="results_gliner_testing.csv",
    output_csv="llama3_1_8B_output.csv"
):

    print(f"Reading: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = [
        "GroupId",
        "fullsummary_gliner"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )

    results = []

    print("Processing summaries...")

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df)
    ):

        group_id = row.GroupId
        text = row.fullsummary_gliner

        try:

            summary = summarize_text(text)

            tone = classify_tone(text)

            results.append({
                "GroupId": group_id,
                "FinalSummary": summary,
                "Tone": tone
            })

        except Exception as e:

            print(f"Error processing GroupId {group_id}: {e}")

            results.append({
                "GroupId": group_id,
                "FinalSummary": "",
                "Tone": "Error"
            })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        output_csv,
        index=False
    )

    print(f"\n✅ Saved: {output_csv}")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    process_csv()