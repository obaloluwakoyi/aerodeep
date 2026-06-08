"""
milestone1/tests/test_stream_b.py
Tests for Stream B NLP pipeline: cleaning, embedding.
"""
import numpy as np
import pytest

from milestone1.stream_b.log_cleaner import LogCleaner


class TestLogCleaner:
    def setup_method(self):
        self.cleaner = LogCleaner()

    def test_abbreviation_expansion(self):
        raw = "LP CYL VIB high, HP BRG temp OOL"
        result = self.cleaner.clean(raw, "test.txt", "shift_log")
        assert "low pressure" in result.cleaned_text.lower()
        assert "high pressure" in result.cleaned_text.lower()
        assert "vibration" in result.cleaned_text.lower()
        assert "bearing" in result.cleaned_text.lower()
        assert "out of limits" in result.cleaned_text.lower()

    def test_fault_extraction(self):
        raw = (
            "Fault: high vibration on HP bearing detected during shift. "
            "Cause: suspected lube oil starvation."
        )
        result = self.cleaner.clean(raw, "test.txt", "maintenance_report")
        assert len(result.fault_descriptions) > 0

    def test_action_extraction(self):
        raw = "Action taken: replaced HP bearing, checked lube oil level."
        result = self.cleaner.clean(raw, "test.txt", "maintenance_report")
        assert len(result.actions_taken) > 0

    def test_unit_id_extraction(self):
        raw = "Work Order for Unit C1001A — HP cylinder inspection"
        result = self.cleaner.clean(raw, "test.txt", "maintenance_report")
        assert result.unit_id == "C1001A"

    def test_component_mention_extraction(self):
        raw = "Intercooler outlet temperature rising. Bearing housing crack found."
        result = self.cleaner.clean(raw, "test.txt", "shift_log")
        comps = result.component_mentions
        assert any("bearing" in c for c in comps)
        assert any("intercooler" in c for c in comps)

    def test_ocr_artefact_removal(self):
        raw = "LP\x84 press\x00ure  spike   detected\n\n\n\nOH  required"
        result = self.cleaner.clean(raw, "test.txt", "shift_log")
        assert "\x84" not in result.cleaned_text
        assert "\x00" not in result.cleaned_text

    def test_date_extraction(self):
        raw = "Shift date: 15/03/2024. HP valve inspection completed."
        result = self.cleaner.clean(raw, "test.txt", "shift_log")
        assert result.date_mentioned is not None
        assert "15" in result.date_mentioned

    def test_empty_document(self):
        result = self.cleaner.clean("", "empty.txt", "shift_log")
        assert result.cleaned_text == ""
        assert result.fault_descriptions == []
        assert result.actions_taken == []
