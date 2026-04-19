# 🧹 Open-Ended Response Text Sanitizer

This project provides a Python script to **sanitize open-ended survey responses** by removing:

* Personal names (e.g., *John Smith, Shannon*)
* Titles (e.g., *Dr., Mr., Professor*)
* Locations (cities, countries)
* Events
* Numbers and special characters
* Possessive forms (e.g., *student’s → student*)

The output is a clean dataset containing:

* `GroupId`
* `sanitized_text`

---

## 📌 Use Case

This script is designed for:

* Course evaluation data
* Survey responses
* Any dataset requiring **PII removal / text normalization**
* Preparing text for **LLMs, NLP analysis, or reporting**

---

## ⚙️ Requirements

* Python 3.8+
* pip

### Install dependencies

```bash
pip install spacy pandas
python -m spacy download en_core_web_sm
```

---

## 📂 Project Structure

```
project-folder/
│
├── process_data_removeppe.py   # Main script
├── OpenEndedResponse5227.csv  # Input file
├── results.csv                # Output file (generated)
└── README.md
```

---

## 📥 Input File Requirements

Your CSV must include:

| Column Name   | Description                       |
| ------------- | --------------------------------- |
| `GroupId`     | Identifier for grouping responses |
| `FullSummary` | Text field to sanitize            |

Example:

```
GroupId,FullSummary
101,"Dr. John Doe visited New York in 2024."
102,"Shannon submitted the report!!!"
```

---

## ▶️ How to Run

From the project directory:

bash
python process_data_removeppe.py


## 📤 Output

A new file will be generated:
results.csv

With structure:

| GroupId | sanitized_text   |
| ------- | ---------------- |
| 101     | visited          |


## 🧠 What the Script Does

### 1. Preprocessing

* Removes numbers
* Removes punctuation and symbols
* Normalizes whitespace

### 2. Title-Based Removal

Removes patterns like:

* `Dr. John Smith`
* `Prof. Adams`
* `Mr. Johnson`

### 3. Named Entity Removal (via spaCy)Bo

Removes:

* `PERSON`
* `GPE` (cities, countries)
* `LOC`
* `EVENT`

### 4. Fallback Name Removal

Removes standalone capitalized words (e.g., *Shannon, Kelly, Bob*) if not detected by spaCy.

### 5. Final Cleanup

* Removes extra spaces
* Ensures clean, readable output

---

## ⚠️ Known Limitations

* May remove non-name capitalized words (e.g., *Excel, Monday*)
* spaCy may miss some edge-case entities
* Aggressive cleaning may reduce sentence readability

---

## 💡 Future Improvements

* Replace removed entities with placeholders (`[PERSON]`, `[LOCATION]`)
* Use `nlp.pipe()` for large datasets (performance optimization)
* Add dictionary-based name detection
* Export structured entity data alongside sanitized text

---

## 🛠️ Troubleshooting

### ❌ File not found

Ensure the CSV file is in the same directory or use a full path.

### ❌ spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### ❌ pip not recognized (Windows)

```bash
python -m pip install spacy pandas
```

---

## 📜 License

This project is open for academic use to be used with LLMs

---

## 👤 Author

Maintained for text processing and NLP workflows involving survey/open-ended response data.
