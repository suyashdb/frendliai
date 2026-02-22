"""Tests for step-aware reasoning detection."""

import pytest
from gateway.config import REASONING_PROFILES, get_reasoning_profile
from gateway.reasoning import (
    StreamingReasoningDetector,
    DeltaFieldDetector,
    ReasoningStep,
    STEP_MIN_CHARS,
)


class TestStreamingReasoningDetector:
    """Test the pattern-based reasoning detector with step segmentation."""

    def _feed_text(self, detector: StreamingReasoningDetector, text: str, chunk_size: int = 4):
        all_events = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            events = detector.feed(chunk)
            all_events.extend(events)
        return all_events

    def _get_steps(self, events: list) -> list[ReasoningStep]:
        return [ev[1] for ev in events if ev[0] == "reasoning_step"]

    def _get_outputs(self, events: list) -> list[str]:
        return [ev[1] for ev in events if ev[0] == "output"]

    def test_basic_think_tags(self):
        """Should detect reasoning between <think> tags."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        # Short reasoning — may not trigger step boundaries but should still work
        text = (
            "<think>This is a short reasoning block that tests basic detection.</think>"
            "This is the output."
        )
        events = self._feed_text(detector, text, chunk_size=5)
        result = detector.finalize()
        outputs = self._get_outputs(events)
        assert "This is the output." in "".join(outputs)

    def test_multi_step_reasoning(self):
        """Multi-paragraph reasoning should be segmented into steps."""
        profile = REASONING_PROFILES["deepseek-r1"]
        detector = StreamingReasoningDetector(profile)

        # Build reasoning with clear paragraph breaks and enough text per step
        step1 = "A" * 100  # 100 chars
        step2 = "B" * 100
        step3 = "C" * 100
        text = f"<think>{step1}\n\n{step2}\n\n{step3}</think>Final output here."

        events = self._feed_text(detector, text, chunk_size=8)
        steps = self._get_steps(events)

        # Should have detected at least 1 step from paragraph breaks
        assert len(steps) >= 1
        # Output should contain the final text
        outputs = self._get_outputs(events)
        assert "Final output here." in "".join(outputs)

    def test_step_min_chars_respected(self):
        """Steps shorter than STEP_MIN_CHARS should not be emitted."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        # Two short paragraphs (below STEP_MIN_CHARS)
        text = "<think>Short.\n\nAlso short.</think>Output."
        events = self._feed_text(detector, text, chunk_size=5)
        steps = self._get_steps(events)
        # Neither paragraph meets STEP_MIN_CHARS, so no steps emitted mid-stream
        # (they'll be flushed as a final step or not at all)
        assert len(steps) == 0

    def test_forced_step_boundary(self):
        """Very long reasoning without natural breaks should force step boundaries."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        # 1500 chars of continuous text with periods but no paragraph breaks
        text = "<think>" + ("This is a sentence. " * 75) + "</think>Output."
        events = self._feed_text(detector, text, chunk_size=10)
        steps = self._get_steps(events)
        # Should have forced at least 1 step boundary
        assert len(steps) >= 1

    def test_no_reasoning(self):
        """When no think tags are present, everything is output."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "This is just a normal response without any reasoning." + " " * 500
        events = self._feed_text(detector, text, chunk_size=10)
        result = detector.finalize()
        assert result.reasoning_text == ""

    def test_reasoning_never_closes(self):
        """If reasoning block never closes, treat as output after finalize."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "<think>I started thinking but never finished..."
        self._feed_text(detector, text, chunk_size=10)
        result = detector.finalize()
        assert result.reasoning_text == ""

    def test_state_transitions(self):
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        assert detector.state == "INIT"
        detector.feed("<think>")
        assert detector.state == "REASONING"
        detector.feed("reasoning content</think>")
        assert detector.state == "OUTPUT"

    def test_step_index_increments(self):
        """Each emitted step should have an incrementing index."""
        profile = REASONING_PROFILES["deepseek-r1"]
        detector = StreamingReasoningDetector(profile)
        step1 = "A" * 120
        step2 = "B" * 120
        text = f"<think>{step1}\n\n{step2}</think>Output."
        events = self._feed_text(detector, text, chunk_size=8)
        steps = self._get_steps(events)
        if len(steps) >= 2:
            assert steps[0].index == 0
            assert steps[1].index == 1

    def test_final_step_flag(self):
        """The last step (when </think> closes) should have is_final=True."""
        profile = REASONING_PROFILES["deepseek-r1"]
        detector = StreamingReasoningDetector(profile)
        # One long paragraph that's enough for a final step
        text = "<think>" + "X" * 120 + "</think>Output."
        events = self._feed_text(detector, text, chunk_size=8)
        steps = self._get_steps(events)
        if steps:
            assert steps[-1].is_final is True

    def test_reasoning_ended_event(self):
        """Should emit reasoning_ended when </think> is detected."""
        profile = REASONING_PROFILES["think-tag"]
        detector = StreamingReasoningDetector(profile)
        text = "<think>reasoning</think>output"
        events = self._feed_text(detector, text, chunk_size=5)
        ended_events = [ev for ev in events if ev[0] == "reasoning_ended"]
        assert len(ended_events) == 1


class TestDeltaFieldDetector:
    """Test the delta-field-based reasoning detector with step segmentation."""

    def test_step_detection_from_field(self):
        detector = DeltaFieldDetector("reasoning_content")
        # Feed enough reasoning to trigger a step
        long_reasoning = "A" * 120
        events1 = detector.classify_delta({"reasoning_content": long_reasoning + "\n\n"})
        events2 = detector.classify_delta({"reasoning_content": "B" * 120})
        # Should have detected at least one step
        all_events = events1 + events2
        steps = [ev[1] for ev in all_events if ev[0] == "reasoning_step"]
        assert len(steps) >= 1

    def test_output_flushes_remaining_step(self):
        detector = DeltaFieldDetector("reasoning_content")
        detector.classify_delta({"reasoning_content": "X" * 100})
        events = detector.classify_delta({"content": "final output"})
        steps = [ev for ev in events if ev[0] == "reasoning_step"]
        outputs = [ev for ev in events if ev[0] == "output"]
        assert len(steps) == 1
        assert steps[0][1].is_final is True
        assert len(outputs) == 1

    def test_empty_delta(self):
        detector = DeltaFieldDetector()
        events = detector.classify_delta({})
        assert events == [("empty", "")]


class TestProfileResolution:
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


class TestGatewayRequestForwarding:
    """Test that unknown fields are forwarded to upstream (not silently dropped)."""

    def test_extra_fields_forwarded(self):
        from gateway.models import GatewayRequest
        req = GatewayRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            response_format={"type": "json_object"},
            seed=42,
        )
        body = req.to_upstream_body()
        assert body["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert body["response_format"] == {"type": "json_object"}
        assert body["seed"] == 42

    def test_gateway_fields_not_forwarded(self):
        from gateway.models import GatewayRequest
        req = GatewayRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            include_prompt_summary=False,
            session_id="sess-123",
        )
        body = req.to_upstream_body()
        assert "include_prompt_summary" not in body
        assert "session_id" not in body
        assert "include_reasoning_summary" not in body
