import spacy
import pandas as pd
import re
import os

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not installed.\nRun: python -m spacy download en_core_web_sm"
    )


# -----------------------------
# CLEAN_TEXT IS FOR MANY USE CASES FOR SPACES AND A NUMBER OF USE CASES RELATED TO THE TEXTS
# -----------------------------
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    # --- Basic cleanup first ---
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\b\w+'s\b", "", text)  # possessive 's
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace

    doc = nlp(text)

    spans_to_remove = []

    # -----------------------------
    # REMOVE TITLES + FOLLOWING NAME TOKENS, THIS USE CASE WOULD BE FOR SOMETHING LIKE DR. SMITH VERSUS SMITH OR DR. JANE DOE
    # -----------------------------
    title_tokens = {
        "dr", "mr", "mrs", "ms", "prof", "professor",
        "sir", "madam", "ta"
    }

    i = 0
    while i < len(doc):
        token_clean = doc[i].text.lower().replace(".", "")

        if token_clean in title_tokens:
            start = doc[i].idx

            # remove title + next 1–2 tokens (name)
            end_index = min(i + 3, len(doc))
            end = doc[end_index - 1].idx + len(doc[end_index - 1])

            spans_to_remove.append((start, end))

            i += 3
            continue

        i += 1

    # -----------------------------
    # REMOVE PERSON / LOCATION / EVENT ENTITIES
    # -----------------------------
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "LOC", "EVENT"]:
            spans_to_remove.append((ent.start_char, ent.end_char))

    spans_to_remove = sorted(spans_to_remove, key=lambda x: x[0], reverse=True)

    for start, end in spans_to_remove:
        text = text[:start] + text[end:]

    # Final cleanup
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered_words = []

    for w in words:
        # removes standalone capitalized words not caught by spaCy
        if w.istitle() and len(w) > 2:
            continue
        filtered_words.append(w)

    text = " ".join(filtered_words)
    return text


# -----------------------------
# PROCESS EACH ROW
# -----------------------------
def process_row(group_id, text):
    return {
        "GroupId": group_id,
        "sanitized_text": clean_text(text)
    }

def process_csv(input_csv, group_column, text_column, output_csv="results.csv"):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    results = []

    for _, row in df.iterrows():
        results.append(process_row(row[group_column], row[text_column]))

    result_df = pd.DataFrame(results)

    if not output_csv.endswith(".csv"):
        output_csv += ".csv"

    result_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")

    return result_df

if __name__ == "__main__":
    process_csv(
        "OpenEndedResponse5227.csv",
        group_column="GroupId",
        text_column="FullSummary",
        output_csv="results.csv"
    )
