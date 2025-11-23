# Geography Exam-Syllabus Alignment Analysis

This project analyzes O-Level Geography exam questions and evaluates their alignment with the official Upper Secondary Geography syllabus using LLM-powered semantic analysis.

## Overview

The system consists of three main components:
1. **Question Extraction** - Extracts questions from exam PDFs
2. **Syllabus Extraction** - Extracts text from syllabus PDFs (with OCR fallback)
3. **Alignment Analysis** - Uses GPT-4 and semantic search to evaluate how well exam questions align with syllabus topics

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- macOS, Linux, or Windows

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to the project directory
cd "/submission/section-2-case-study/codes"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter issues with `pytesseract`, you may need to install Tesseract OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Configure OpenAI API Key

Open `03_run_syllabus_topic_alignment_poc.py` and add your OpenAI API key:

```python
# Line 14 in 03_run_syllabus_topic_alignment_poc.py
OPENAI_API_KEY = "your-openai-api-key-here"
```

**Important:** Replace `"your-openai-api-key-here"` with your actual OpenAI API key.

## Project Structure

```
codes/
├── data/                                    # Input PDFs
│   ├── 2025 O-Level Geography Paper.pdf
│   └── Upper Secondary Geography Syllabus.pdf
├── output/                                  # Generated outputs
│   ├── geography_questions.json             # Extracted questions
│   ├── syllabus_text.txt                    # Extracted syllabus text
│   ├── geography_llm_alignment_results.json # Detailed analysis results
│   └── geography_llm_alignment_report.md    # Human-readable report
├── 01_extract_questions_poc.py              # Question extraction script
├── 02_extract_syllabus_poc.py               # Syllabus extraction script
├── 03_run_syllabus_topic_alignment_poc.py   # Main alignment analysis
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## Usage

### Step 1: Extract Questions from Exam Paper

```bash
python 01_extract_questions_poc.py
```

**Output:** `output/geography_questions.json`

This script:
- Reads the exam PDF using `pdfplumber`
- Identifies question boundaries
- Cleans and structures the extracted text
- Saves questions as JSON

### Step 2: Extract Syllabus Text

```bash
python 02_extract_syllabus_poc.py
```

**Output:** `output/syllabus_text.txt`

This script:
- Extracts text from the syllabus PDF using PyMuPDF
- Falls back to OCR for pages with minimal extractable text
- Saves the complete syllabus text

### Step 3: Run Alignment Analysis

```bash
python 03_run_syllabus_topic_alignment_poc.py
```

**Outputs:**
- `output/geography_llm_alignment_results.json` - Detailed JSON results
- `output/geography_llm_alignment_report.md` - Human-readable markdown report

This script:
- Loads extracted questions and syllabus text
- Chunks the syllabus into manageable pieces
- Creates embeddings using OpenAI's `text-embedding-3-small`
- Builds a FAISS index for semantic search
- For each question:
  - Retrieves top-5 most relevant syllabus chunks
  - Uses GPT-4 to analyze alignment with syllabus
  - Identifies matched clusters, topics, and assessment objectives
  - Assigns alignment scores (0-5 scale)
- Generates comprehensive reports with topic weightage analysis

## Configuration Options

You can adjust these parameters in `03_run_syllabus_topic_alignment_poc.py`:

```python
OPENAI_MODEL = "gpt-4o"                # GPT model for analysis
EMBED_MODEL = "text-embedding-3-small" # Embedding model
CHUNK_SIZE = 900                       # Characters per syllabus chunk
CHUNK_OVERLAP = 150                    # Overlap between chunks
TOP_K = 5                              # Number of chunks retrieved per question
```

For `02_extract_syllabus_poc.py`:

```python
OCR_ENABLED = True     # Enable/disable OCR fallback
OCR_THRESHOLD = 40     # Minimum chars before triggering OCR
OCR_DPI = 200          # OCR resolution (higher = better quality, slower)
```

## Output Format

### Alignment Results JSON

```json
{
  "per_question": [
    {
      "question_number": 1,
      "question_text": "...",
      "matched_clusters": ["Cluster 1: Geography in Everyday Life"],
      "matched_topics": ["Cluster 1 Topic 1.3: Geographical Methods"],
      "assessment_objectives": ["AO1", "AO2"],
      "alignment_score": 4.5,
      "alignment_strength": "Strong",
      "alignment_explanation": "...",
      "retrieved_chunks": [...]
    }
  ],
  "summary": {
    "total_questions": 3,
    "cluster_weightage": {...},
    "topic_weightage": {...},
    "assessment_objectives_distribution": {...}
  }
}
```

### Markdown Report

The report includes:
- Per-question alignment analysis
- Cluster and topic weightage tables
- Assessment objectives distribution
- Interpretation guidelines

## Troubleshooting

### Issue: "No module named 'pdfplumber'"
**Solution:** Ensure you've activated the virtual environment and installed requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"
**Solution:** Make sure you've set your API key in `03_run_syllabus_topic_alignment_poc.py` (line 14)

### Issue: OCR not working
**Solution:** Install Tesseract OCR system package (see Prerequisites section)

### Issue: "FAISS not found" or installation errors
**Solution:** Try installing FAISS-CPU explicitly:
```bash
pip install faiss-cpu
```

## Cost Estimation

Running the full pipeline on 3 questions approximately costs:
- **Embeddings**: ~$0.01 (syllabus chunks + questions)
- **GPT-4 calls**: ~$0.15-0.30 (depending on question length)
- **Total**: ~$0.20-0.35 per run

## Notes

- The scripts are designed for the specific PDF formats provided
- OCR fallback ensures robust text extraction from image-based PDFs
- The alignment analysis uses GPT-4's reasoning capabilities to evaluate syllabus coverage
- Results are deterministic with `temperature=0.1` for reproducibility

## License

This is a proof-of-concept project for GovTech assessment purposes.

## Support

For issues or questions, please refer to the code comments or contact the development team.

