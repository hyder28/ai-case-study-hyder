import pdfplumber
import re
import json

# ---------------------------------------------------------
# Step 1 — Read PDF with pdfplumber
# ---------------------------------------------------------

pdf_path = "./data/2025 O-Level Geography Paper.pdf"

print("Extracting text using pdfplumber...")
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n\n"

raw_text = text

# ---------------------------------------------------------
# Step 2 — Clean the extracted text
# ---------------------------------------------------------

cleaned = raw_text

# Normalize whitespace
cleaned = cleaned.replace("\r", "")
cleaned = re.sub(r"\n{2,}", "\n", cleaned)

# Remove dotted answer lines (ASCII + Unicode)
cleaned = re.sub(
    r"(?m)^[\.\-–—_•·\u2024\u2025\u2026\u00b7\s]{4,}$",
    "",
    cleaned
)

cleaned = re.sub(
    r"[\.\-–—_•·\u2024\u2025\u2026\u00b7]{4,}",
    "",
    cleaned
)

# Remove © footers, blank page, etc.
cleaned = re.sub(r"©.*?(?=\n)", "", cleaned)
cleaned = cleaned.replace("BLANK PAGE", "")
cleaned = cleaned.replace("[Turn over]", "")

# Remove standalone page numbers
cleaned = re.sub(r"(?m)^\s*\d+\s*$", "", cleaned)

cleaned = cleaned.strip()

# ---------------------------------------------------------
# Step 3 — Detect question blocks (REAL FORMAT)
# ---------------------------------------------------------
# From pdfplumber output, questions appear as:
#   "1  Cluster 1: Geography in Everyday Life"
#   "2  Cluster 2: Tourism"
#   "3  Cluster 3: Climate"
#
# So pattern is: number + whitespace + 'Cluster'

pattern = r"(?m)^\s*(\d+)\s+Cluster"

matches = list(re.finditer(pattern, cleaned))

if not matches:
    print("DEBUG: Could not find question boundaries.")
    print(cleaned[:800])
    raise ValueError("Could not detect question boundaries — update regex.")

question_blocks = []

for i in range(len(matches)):
    start = matches[i].start()
    end   = matches[i+1].start() if i+1 < len(matches) else len(cleaned)
    block = cleaned[start:end].strip()
    question_blocks.append(block)

print(f"Extracted {len(question_blocks)} question blocks.")


def clean_block(q):
    q = re.sub(r"[ \t]{2,}", " ", q)
    q = q.replace("\u0007", "")
    q = q.strip()
    return q

question_blocks = [clean_block(q) for q in question_blocks]


for i, q in enumerate(question_blocks, 1):
    print("\n" + "="*70)
    print(f"CLEAN QUESTION {i}")
    print("="*70)
    print(q[:600], "...")
    print()


output = [
    {"question_number": i+1, "text": q}
    for i, q in enumerate(question_blocks)
]

with open("./output/geography_questions.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to geography_questions_clean.json")
