import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# ========= CONFIG =========
PDF_PATH = "./data/Upper Secondary Geography Syllabus.pdf"
OUTPUT_PATH = "./output/syllabus_text.txt"

OCR_ENABLED = True          # Enable fallback OCR
OCR_THRESHOLD = 40          # If extracted text < this, OCR triggers
OCR_DPI = 200               # Higher -> better OCR, slower


def extract_pdf_text_pymupdf(pdf_path, ocr_threshold=40, ocr=True):
    """
    Extract text from a PDF using PyMuPDF.
    If a page has very little extractable text, run OCR fallback.
    This method works well for MOE/CAIE syllabus PDFs on macOS CPU.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    print("Starting syllabus extraction...")
    print(f"PDF: {pdf_path}\n")

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        text = text.replace("\x00", "").replace("\x08", "").strip()

        if len(text) < ocr_threshold and ocr:
            print(f"  • Page {i+1}: Very little text ({len(text)} chars) → OCR fallback")

            # Render page as image
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes()))

            ocr_text = pytesseract.image_to_string(img)
            ocr_text = ocr_text.replace("\x00", "").strip()
            text = ocr_text
        else:
            print(f"  • Page {i+1}: Extracted {len(text)} chars via PyMuPDF")

        if text:
            all_text.append(text)

    full_text = "\n\n".join(all_text)
    print(f"\nExtraction complete. Total characters extracted: {len(full_text)}")
    return full_text


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Cannot find PDF: {PDF_PATH}")

    syllabus_text = extract_pdf_text_pymupdf(
        PDF_PATH,
        ocr_threshold=OCR_THRESHOLD,
        ocr=OCR_ENABLED
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(syllabus_text)

    print(f"\n✔ Syllabus text saved to: {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
