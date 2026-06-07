"""
milestone1/stream_b/log_cleaner.py

Cleans and normalises raw extracted text from maintenance logs.
Handles offshore/compressor-domain abbreviations, OCR artefacts,
date normalisation, and structured segment extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ── Domain abbreviation dictionary ───────────────────────────────────────────
# Offshore compressor / mechanical engineering abbreviations
OFFSHORE_ABBREVIATIONS: Dict[str, str] = {
    # Pressure / temperature
    r"\bLP\b": "low pressure",
    r"\bHP\b": "high pressure",
    r"\bIP\b": "intermediate pressure",
    r"\bDP\b": "differential pressure",
    r"\bPSIG\b": "psi gauge",
    r"\bPSIA\b": "psi absolute",
    r"\bBARG\b": "bar gauge",
    r"\bTEMP\b": "temperature",
    r"\bT/C\b": "thermocouple",
    # Mechanical
    r"\bVIB\b": "vibration",
    r"\bBRG\b": "bearing",
    r"\bCYL\b": "cylinder",
    r"\bPKG\b": "packing",
    r"\bVLV\b": "valve",
    r"\bSEAL GAS\b": "seal gas",
    r"\bSGS\b": "seal gas system",
    r"\bDGS\b": "dry gas seal",
    r"\bMCS\b": "mechanical contact seal",
    r"\bLO\b": "lube oil",
    r"\bLOP\b": "lube oil pressure",
    r"\bLOT\b": "lube oil temperature",
    r"\bLOL\b": "lube oil level",
    r"\bXBD\b": "crosshead bearing",
    r"\bFPD\b": "foundation bolt",
    r"\bCOD\b": "carbon deposit",
    # Maintenance actions
    r"\bPM\b": "preventive maintenance",
    r"\bCM\b": "corrective maintenance",
    r"\bOH\b": "overhaul",
    r"\bInsp\b": "inspection",
    r"\bRpl\b": "replace",
    r"\bR/R\b": "remove and replace",
    r"\bO/H\b": "overhaul",
    r"\bT/D\b": "turnaround",
    r"\bWO\b": "work order",
    # Status
    r"\bN\/A\b": "not applicable",
    r"\bN\/F\b": "no fault",
    r"\bAFD\b": "anomaly found during",
    r"\bOOL\b": "out of limits",
    r"\bHHH\b": "very high alarm",
    r"\bHH\b": "high high alarm",
    r"\bLL\b": "low low alarm",
}


@dataclass
class CleanedDocument:
    """Cleaned and segmented document ready for embedding."""
    source_path: str
    document_type: str
    raw_text: str
    cleaned_text: str
    # Structured segments extracted from the document
    fault_descriptions: List[str]
    actions_taken: List[str]
    component_mentions: List[str]
    date_mentioned: Optional[str]
    unit_id: Optional[str]


class LogCleaner:
    """
    Full text cleaning and structured segment extraction pipeline
    for offshore maintenance documents.
    """

    # Patterns for structured extraction
    FAULT_PATTERNS = [
        r"(?:fault|failure|defect|problem|issue|anomaly|alarm|trip|abnormal)[:\s]+([^.\n]+)",
        r"(?:found|observed|noticed|detected)[:\s]+([^.\n]+(?:leak|wear|damage|crack|vibrat|noise|high|low|excess)[^.\n]*)",
        r"(?:cause|root cause|reason)[:\s]+([^.\n]+)",
    ]

    ACTION_PATTERNS = [
        r"(?:action|corrective action|action taken|repair|replaced|installed|adjusted|cleaned|inspected)[:\s]+([^.\n]+)",
        r"(?:recommendation|next step|plan)[:\s]+([^.\n]+)",
    ]

    COMPONENT_PATTERNS = [
        r"\b(?:low.pressure|high.pressure|LP|HP|intercooler|bearing|cylinder|valve|"
        r"seal|packing|coupling|shaft|impeller|piston|rod|crosshead|frame|"
        r"lube oil|cooler|separator|scrubber|suction|discharge)\b",
    ]

    UNIT_ID_PATTERNS = [
        r"(?:unit|compressor|tag|equipment)\s*(?:no\.?|number|id|#)?\s*[:\s]?\s*([A-Z]{1,4}[-_]?\d{3,6})",
        r"\b([CK]\d{3,5}[A-Z]?)\b",   # e.g. C1001A
    ]

    DATE_PATTERNS = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    ]

    def __init__(self, extra_abbreviations: Optional[Dict[str, str]] = None):
        self._abbrevs = dict(OFFSHORE_ABBREVIATIONS)
        if extra_abbreviations:
            self._abbrevs.update(extra_abbreviations)
        # Compile patterns
        self._abbrev_compiled = [
            (re.compile(pat, re.IGNORECASE), replacement)
            for pat, replacement in self._abbrevs.items()
        ]
        self._fault_re = [re.compile(p, re.IGNORECASE) for p in self.FAULT_PATTERNS]
        self._action_re = [re.compile(p, re.IGNORECASE) for p in self.ACTION_PATTERNS]
        self._component_re = [re.compile(p, re.IGNORECASE) for p in self.COMPONENT_PATTERNS]
        self._unit_id_re = [re.compile(p, re.IGNORECASE) for p in self.UNIT_ID_PATTERNS]
        self._date_re = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]

    def clean(self, raw_text: str, source_path: str, document_type: str) -> CleanedDocument:
        cleaned = self._clean_text(raw_text)

        return CleanedDocument(
            source_path=source_path,
            document_type=document_type,
            raw_text=raw_text,
            cleaned_text=cleaned,
            fault_descriptions=self._extract_faults(cleaned),
            actions_taken=self._extract_actions(cleaned),
            component_mentions=self._extract_components(cleaned),
            date_mentioned=self._extract_date(raw_text),
            unit_id=self._extract_unit_id(raw_text),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        # Remove OCR artefacts
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # Remove page break markers
        text = re.sub(r"---\s*PAGE BREAK\s*---", "\n", text)

        # Collapse excessive whitespace
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lone numbers/codes that are likely artefacts
        text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

        # Expand abbreviations
        for pattern, replacement in self._abbrev_compiled:
            text = pattern.sub(replacement, text)

        # Normalise common OCR digit/letter confusions
        text = re.sub(r"\b0il\b", "oil", text, flags=re.IGNORECASE)

        return text.strip()

    def _extract_faults(self, text: str) -> List[str]:
        faults = []
        for pattern in self._fault_re:
            for match in pattern.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) > 10:
                    faults.append(snippet)
        return list(dict.fromkeys(faults))[:10]  # deduplicate, cap at 10

    def _extract_actions(self, text: str) -> List[str]:
        actions = []
        for pattern in self._action_re:
            for match in pattern.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) > 10:
                    actions.append(snippet)
        return list(dict.fromkeys(actions))[:10]

    def _extract_components(self, text: str) -> List[str]:
        found = set()
        for pattern in self._component_re:
            for match in pattern.finditer(text):
                found.add(match.group(0).lower().strip())
        return sorted(found)

    def _extract_unit_id(self, text: str) -> Optional[str]:
        for pattern in self._unit_id_re:
            m = pattern.search(text)
            if m:
                return m.group(1).upper()
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        for pattern in self._date_re:
            m = pattern.search(text)
            if m:
                return m.group(1)
        return None
