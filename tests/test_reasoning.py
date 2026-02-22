"""Tests for reasoning detection."""

import pytest
from gateway.config import REASONING_PROFILES, get_reasoning_profile
from gateway.reasoning import StreamingReasoningDetector, DeltaFieldDetector


class TestStreamingReasoningDetector:
    """Test the pattern-based reasoning detector."""

    def _feed_text(self, detector: StreamingReasoningDetector, text: str, chunk_size: int = 4):
        """Feed text in small chunks, like real streaming."""
        all_events = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            events = detector.feed(chunk)
            all_events.extend(events)
        return all_events

    def test_basic_think_tags(self):
        """Should detect reasoning between <think> tags."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "<think>This is reasoning.</think>This is the output."
        events = self._feed_text(detector, text, chunk_size=5)

        result = detector.finalize()
        assert result.reasoning_text.strip() == "This is reasoning."
        assert "This is the output." in "".join(result.output_tokens)

    def test_no_reasoning(self):
        """When no think tags are present, everything is output."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "This is just a normal response without any reasoning."

        # Feed enough text to trigger the 200-char fallback
        padding = " " * 200
        events = self._feed_text(detector, text + padding, chunk_size=10)
        result = detector.finalize()

        assert result.reasoning_text == ""
        assert text in "".join(result.output_tokens)

    def test_multiline_reasoning(self):
        """Reasoning can span multiple lines."""
        profile = REASONING_PROFILES["deepseek-r1"]
        detector = StreamingReasoningDetector(profile)
        text = (
            "<think>\n"
            "Step 1: Analyze the problem\n"
            "Step 2: Consider alternatives\n"
            "Step 3: Conclude\n"
            "</think>\n"
            "Here is the final answer."
        )
        self._feed_text(detector, text, chunk_size=8)
        result = detector.finalize()

        assert "Step 1" in result.reasoning_text
        assert "Step 3" in result.reasoning_text
        assert "final answer" in "".join(result.output_tokens)

    def test_reasoning_never_closes(self):
        """If reasoning block never closes, treat as output after finalize."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "<think>I started thinking but never finished..."
        self._feed_text(detector, text, chunk_size=10)
        result = detector.finalize()

        # Should not have reasoning since it never closed
        assert result.reasoning_text == ""

    def test_state_transitions(self):
        """Verify state machine transitions."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)

        assert detector.state == "INIT"
        detector.feed("<think>")
        assert detector.state == "REASONING"
        detector.feed("reasoning content</think>")
        assert detector.state == "OUTPUT"


class TestDeltaFieldDetector:
    """Test the delta-field-based reasoning detector."""

    def test_separate_fields(self):
        detector = DeltaFieldDetector("reasoning_content")

        cls, text = detector.classify_delta({"reasoning_content": "thinking..."})
        assert cls == "reasoning"
        assert text == "thinking..."

        cls, text = detector.classify_delta({"content": "output text"})
        assert cls == "output"
        assert text == "output text"

        result = detector.finalize()
        assert result.reasoning_text == "thinking..."
        assert "output text" in "".join(result.output_tokens)

    def test_empty_delta(self):
        detector = DeltaFieldDetector()
        cls, text = detector.classify_delta({})
        assert cls == "empty"
        assert text == ""


class TestProfileResolution:
    """Test model → profile mapping."""

    def test_deepseek_r1(self):
        p = get_reasoning_profile("deepseek-r1-distill-llama-70b")
        assert p.name == "deepseek-r1"

    def test_qwq(self):
        p = get_reasoning_profile("Qwen/QwQ-32B")
        assert p.name == "qwen-qwq"

    def test_unknown_model_gets_default(self):
        p = get_reasoning_profile("some-random-model-v2")
        assert p.name == "think-tag"

    def test_o1_model(self):
        p = get_reasoning_profile("o1-preview")
        assert p.name == "reasoning-field"
        assert p.delta_field == "reasoning_content"
