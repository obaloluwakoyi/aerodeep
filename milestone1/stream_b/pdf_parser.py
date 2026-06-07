"""
milestone1/stream_b/pdf_parser.py

Extracts raw text from maintenance PDFs and scanned shift reports.
Handles both digital-native PDFs (pdfplumber) and scanned images
requiring OCR (pytesseract via PyMuPDF rasterisation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from loguru import logger


@dataclass
class ExtractedDocument:
    """Parsed document with metadata."""
    source_path: str
    document_type: str          # "maintenance_report" | "shift_log" | "inspection"
    extracted_text: str
    page_count: int
    extraction_method: str      # "digital" | "ocr" | "mixed"
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """
    Two-path PDF text extractor.

    Path A — Digital: pdfplumber for text-native PDFs.
             Fast, preserves structure, accurate.
    Path B — OCR:     PyMuPDF rasterises pages → Tesseract OCR.
             Used when digital extraction yields < MIN_TEXT_RATIO
             of expected content (i.e., the PDF is a scan).
    """

    MIN_DIGITAL_CHARS_PER_PAGE = 100
    OCR_DPI = 300

    def __init__(
        self,
        ocr_language: str = "eng",
        ocr_config: str = "--psm 6",
    ):
        self._ocr_lang = ocr_language
        self._ocr_config = ocr_config

    def parse(self, pdf_path: str) -> ExtractedDocument:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_type = self._infer_document_type(path.name)

        # Try digital extraction first
        digital_text, page_count = self._extract_digital(pdf_path)
        avg_chars = len(digital_text) / max(page_count, 1)

        if avg_chars >= self.MIN_DIGITAL_CHARS_PER_PAGE:
            method = "digital"
            text = digital_text
            logger.info(f"Digital extraction: {path.name} ({len(text)} chars)")
        else:
            # Scanned — fall back to OCR
            logger.info(f"Scanned PDF detected, using OCR: {path.name}")
            text = self._extract_ocr(pdf_path, page_count)
            method = "ocr" if not digital_text.strip() else "mixed"
            logger.info(f"OCR extraction: {path.name} ({len(text)} chars)")

        return ExtractedDocument(
            source_path=str(pdf_path),
            document_type=doc_type,
            extracted_text=text,
            page_count=page_count,
            extraction_method=method,
            metadata=self._extract_metadata(pdf_path),
        )

    def _extract_digital(self, pdf_path: str) -> tuple[str, int]:
        """Extract text using pdfplumber (digital-native PDFs)."""
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Also extract tables as structured text
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned_row = [str(cell or "").strip() for cell in row]
                        text += "\n" + " | ".join(cleaned_row)
                pages_text.append(text)

        return "\n\n--- PAGE BREAK ---\n\n".join(pages_text), page_count

    def _extract_ocr(self, pdf_path: str, page_count: int) -> str:
        """Rasterise pages and apply Tesseract OCR."""
        doc = fitz.open(pdf_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Rasterise at target DPI
            mat = fitz.Matrix(self.OCR_DPI / 72, self.OCR_DPI / 72)
            pix = page.get_pixmap(matrix=mat)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Pre-process: convert to greyscale for better OCR
            img_grey = img.convert("L")

            text = pytesseract.image_to_string(
                img_grey,
                lang=self._ocr_lang,
                config=self._ocr_config,
            )
            pages_text.append(text)

        doc.close()
        return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

    @staticmethod
    def _infer_document_type(filename: str) -> str:
        fname = filename.lower()
        if any(kw in fname for kw in ["maint", "pm", "repair", "overhaul"]):
            return "maintenance_report"
        if any(kw in fname for kw in ["shift", "handover", "log", "daily"]):
            return "shift_log"
        if any(kw in fname for kw in ["inspect", "survey", "check"]):
            return "inspection"
        return "general"

    @staticmethod
    def _extract_metadata(pdf_path: str) -> dict:
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            doc.close()
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "creation_date": meta.get("creationDate", ""),
                "modification_date": meta.get("modDate", ""),
            }
        except Exception:
            return {}


class TextFileParser:
    """Parser for plain-text shift logs and operator notes."""

    def parse(self, file_path: str) -> ExtractedDocument:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_type = PDFParser._infer_document_type(path.name)
        return ExtractedDocument(
            source_path=str(file_path),
            document_type=doc_type,
            extracted_text=text,
            page_count=1,
            extraction_method="plaintext",
            metadata={"filename": path.name},
        )
