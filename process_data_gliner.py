from gliner import GLiNER
import pandas as pd
import torch
import re
import os
from tqdm import tqdm

# ---------------------------------------------------
# FORCE LOCAL / OFFLINE MODE
# ---------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"

# ---------------------------------------------------
# MODEL PATH
# ---------------------------------------------------
MODEL_PATH = (
    r"C:\Users\goynej\.cache\huggingface\hub\models--urchade--gliner_medium-v2.1"
    r"\snapshots\40ec419335d09393f298636f471328b722c6da9e"
)

# ---------------------------------------------------
# LOAD GLINER MODEL
# ---------------------------------------------------
print("Loading GLiNER model...")

model = GLiNER.from_pretrained(MODEL_PATH)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move model to GPU if available
model.model.to(device)

print(f"Using device: {device}")

# ---------------------------------------------------
# ENTITY LABELS
# ---------------------------------------------------
labels = [
    "person",
    "student name",
    "instructor name",
    "professor",
    "location",
    "organization",
    "university",
    "email address"
]

# ---------------------------------------------------
# CHUNK TEXT
# Prevents GLiNER truncation warnings
# ---------------------------------------------------
def chunk_text(text, chunk_size=1000):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

# ---------------------------------------------------
# CLEAN USING GLINER
# ---------------------------------------------------
def clean_with_gliner(text):

    if pd.isna(text):
        return ""

    text = str(text)

    chunks = chunk_text(text)

    cleaned_chunks = []

    for chunk in chunks:

        try:
            entities = model.predict_entities(
                chunk,
                labels,
                threshold=0.5
            )

            spans = []

            for ent in entities:
                start = ent["start"]
                end = ent["end"]
                spans.append((start, end))

            # Remove from END → START
            spans = sorted(
                spans,
                key=lambda x: x[0],
                reverse=True
            )

            for start, end in spans:

                # OPTION 1 = REMOVE ENTITY
                # chunk = chunk[:start] + chunk[end:]

                # OPTION 2 = REPLACE ENTITY
                chunk = (
                    chunk[:start]
                    + "[REDACTED]"
                    + chunk[end:]
                )

            # Cleanup spaces
            chunk = re.sub(r"\s+", " ", chunk).strip()

            cleaned_chunks.append(chunk)

        except Exception as e:
            print(f"Error processing chunk: {e}")
            cleaned_chunks.append(chunk)

    return " ".join(cleaned_chunks)

# ---------------------------------------------------
# PROCESS CSV
# ---------------------------------------------------
def process_csv(input_csv, output_csv):

    print(f"Reading: {input_csv}")

    df = pd.read_csv(input_csv)

    # Validate columns
    required_columns = ["GroupId", "sanitized_text"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    output_rows = []

    print("Processing rows with GLiNER...")

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df)
    ):

        group_id = row.GroupId
        text = row.sanitized_text

        cleaned = clean_with_gliner(text)

        output_rows.append({
            "GroupId": group_id,
            "fullsummary_gliner": cleaned
        })

    output_df = pd.DataFrame(output_rows)

    output_df.to_csv(output_csv, index=False)

    print(f"\n✅ GLiNER output saved to: {output_csv}")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    input_csv = "results_output.csv"
    output_csv = "results_gliner.csv"

    process_csv(input_csv, output_csv)