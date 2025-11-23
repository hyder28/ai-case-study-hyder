# HDB Resale Portal Impact Analysis

Analysis of whether the HDB Resale Portal (launched January 2018) reduced business for property agents.

## Setup

### Install Dependencies
```bash
pip install pandas matplotlib jupyter
```

### Required Data Files

Place these CSV files in the same directory as the notebook:

1. **`CEASalespersonsPropertyTransactionRecordsresidential.csv`**
   - CEA agent transaction records

2. **`Resale flat prices based on registration date from Jan-2017 onwards.csv`**
   - HDB resale transactions from data.gov.sg

## Running the Notebook

Navigate to this directory and launch Jupyter:

```bash
cd "/submission/section-1-question-1/codes"
jupyter notebook
```

Open `codes_question1.ipynb` and run all cells.

## Output

The notebook generates:
- Monthly transaction volume analysis
- Agent market share calculations
- Two plots showing trends before/after portal launch

## Troubleshooting

**File not found error:** Ensure both CSV files are in the same directory as the notebook.

**Missing module error:** Run `pip install pandas matplotlib jupyter`
