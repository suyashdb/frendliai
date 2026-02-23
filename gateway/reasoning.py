"""
Reasoning token detector with step-level segmentation.

Consumes streamed deltas from the upstream API and:
  1. Classifies tokens as reasoning vs. output
  2. Segments reasoning into logical "steps" for incremental summarisation

Step boundaries are detected by:
  - Paragraph breaks (double newline)
  - Numbered step markers ("Step 1:", "1.", "First,", etc.)
  - Token count ceiling (force boundary after N chars if no natural break)

This enables the gateway to summarise reasoning incrementally — the client
sees reasoning summaries appearing in real-time while the model still thinks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from gateway.config import ReasoningProfile


# ---------------------------------------------------------------------------
# Step boundary detection
# ---------------------------------------------------------------------------

# Patterns that indicate a new reasoning step is starting
STEP_BOUNDARY_PATTERNS = [
    re.compile(r"\n\n"),                          # Paragraph break
    re.compile(r"\n(?=\d+[\.\)]\s)"),             # "1. ", "2) "
    re.compile(r"\n(?=Step \d)", re.IGNORECASE),  # "Step 1", "Step 2"
    re.compile(r"\n(?=(?:First|Second|Third|Next|Then|Finally|Now|So|Therefore|However|Alternatively|Wait|Let me|Actually|Hmm)[,\s])", re.IGNORECASE),
]

# If no natural boundary detected after this many chars, force a split
STEP_MAX_CHARS = 600

# Minimum chars for a step to be worth summarising
STEP_MIN_CHARS = 80


@dataclass
class ReasoningStep:
    """A single logical step within the model's reasoning."""
    index: int
    text: str
    is_final: bool = False  # True for the last step (when </think> closes)


@dataclass
class DetectionResult:
    """Final output of the reasoning detector after the stream ends."""
    reasoning_text: str
    output_tokens: list[str]
    raw_text: str


class StreamingReasoningDetector:
    """
    Incrementally fed tokens. Detects reasoning boundaries and segments
    reasoning into steps.

    States:
        INIT       -> haven't seen any reasoning marker yet
        REASONING  -> inside a reasoning block, accumulating steps
        OUTPUT     -> past the reasoning block; forwarding output tokens

    Emits events:
        ("reasoning_step", ReasoningStep)  - a completed reasoning step
        ("output", text)                   - an output token to forward
        ("buffered", token)                - token held internally (ambiguity)
        ("reasoning_ended",)               - reasoning block closed
    """

    def __init__(self, profile: ReasoningProfile) -> None:
        self._profile = profile
        self._buffer = ""
        self._reasoning_buffer = ""
        self._reasoning_parts: list[str] = []
        self._output_parts: list[str] = []
        self._state: str = "INIT"
        self._reasoning_ended = False
        self._step_index = 0
        self._pending_step_text = ""

    def feed(self, token: str) -> list[tuple]:
        """
        Feed a new token. Returns a list of typed events.
        """
        self._buffer += token
        events: list[tuple] = []

        if self._state == "OUTPUT":
            self._output_parts.append(token)
            events.append(("output", token))
            return events

        if self._state == "INIT":
            if self._profile.start_pattern.search(self._buffer):
                self._state = "REASONING"
                match = self._profile.start_pattern.search(self._buffer)
                if match and match.start() > 0:
                    pre = self._buffer[:match.start()]
                    self._output_parts.append(pre)
                    events.append(("output", pre))
                if match:
                    after_tag = self._buffer[match.end():]
                    if after_tag:
                        self._pending_step_text += after_tag
                        self._reasoning_buffer += after_tag
                return events
            else:
                if len(self._buffer) > self._profile.no_reasoning_threshold:
                    self._state = "OUTPUT"
                    self._output_parts.append(self._buffer)
                    events.append(("output", self._buffer))
                else:
                    events.append(("buffered", token))
                return events

        if self._state == "REASONING":
            self._reasoning_buffer += token
            self._pending_step_text += token

            # Check if reasoning has ended
            if self._profile.end_pattern.search(self._reasoning_buffer):
                self._state = "OUTPUT"
                self._reasoning_ended = True

                end_match = self._profile.end_pattern.search(self._reasoning_buffer)
                if end_match:
                    # Get final step text up to end marker
                    end_in_pending = self._profile.end_pattern.search(self._pending_step_text)
                    if end_in_pending:
                        final_step_text = self._pending_step_text[:end_in_pending.start()]
                    else:
                        final_step_text = self._pending_step_text

                    final_step_text = self._profile.end_pattern.sub("", final_step_text).strip()

                    if len(final_step_text) >= STEP_MIN_CHARS:
                        step = ReasoningStep(
                            index=self._step_index,
                            text=final_step_text,
                            is_final=True,
                        )
                        self._step_index += 1
                        events.append(("reasoning_step", step))

                    self._pending_step_text = ""

                    r_text = self._reasoning_buffer[:end_match.start()]
                    self._reasoning_parts = [r_text]

                    after = self._reasoning_buffer[end_match.end():]
                    if after:
                        self._output_parts.append(after)
                        events.append(("output", after))

                    events.append(("reasoning_ended",))
                return events

            # Check for step boundaries within ongoing reasoning
            step_events = self._check_step_boundaries()
            events.extend(step_events)
            return events

        return events

    def _check_step_boundaries(self) -> list[tuple]:
        """Check if pending reasoning text contains a step boundary."""
        events = []

        for pattern in STEP_BOUNDARY_PATTERNS:
            match = pattern.search(self._pending_step_text)
            if match:
                step_text = self._pending_step_text[:match.start()].strip()
                remaining = self._pending_step_text[match.end():]

                if len(step_text) >= STEP_MIN_CHARS:
                    step = ReasoningStep(index=self._step_index, text=step_text)
                    self._step_index += 1
                    events.append(("reasoning_step", step))
                    self._pending_step_text = remaining
                    return events

        # Force boundary if accumulated too much text without a natural break
        if len(self._pending_step_text) >= STEP_MAX_CHARS:
            last_period = self._pending_step_text.rfind(".", STEP_MIN_CHARS, STEP_MAX_CHARS)
            last_newline = self._pending_step_text.rfind("\n", STEP_MIN_CHARS, STEP_MAX_CHARS)
            split_at = max(last_period, last_newline)
            if split_at < STEP_MIN_CHARS:
                split_at = STEP_MAX_CHARS

            step_text = self._pending_step_text[:split_at + 1].strip()
            remaining = self._pending_step_text[split_at + 1:]

            if len(step_text) >= STEP_MIN_CHARS:
                step = ReasoningStep(index=self._step_index, text=step_text)
                self._step_index += 1
                events.append(("reasoning_step", step))
                self._pending_step_text = remaining

        return events

    def finalize(self) -> DetectionResult:
        """Call when the upstream stream ends. Flushes remaining content."""
        if self._state == "REASONING":
            reasoning = ""
            output_text = self._buffer
            match = self._profile.start_pattern.search(self._buffer)
            if match:
                output_text = self._buffer[:match.start()] + self._buffer[match.end():]
            self._output_parts = [output_text]
        elif self._state == "INIT":
            self._output_parts = [self._buffer]

        return DetectionResult(
            reasoning_text="".join(self._reasoning_parts),
            output_tokens=self._output_parts,
            raw_text=self._buffer,
        )

    @property
    def has_reasoning(self) -> bool:
        return self._reasoning_ended and bool(self._reasoning_parts)

    @property
    def state(self) -> str:
        return self._state

    @property
    def pending_step_text(self) -> str:
        return self._pending_step_text

    @property
    def step_count(self) -> int:
        return self._step_index


class DeltaFieldDetector:
    """
    For models that emit reasoning in a separate delta field
    (e.g., `reasoning_content`). Includes step segmentation.
    """

    def __init__(self, field_name: str = "reasoning_content") -> None:
        self._field_name = field_name
        self._reasoning_parts: list[str] = []
        self._output_parts: list[str] = []
        self._pending_step_text = ""
        self._step_index = 0

    def classify_delta(self, delta: dict) -> list[tuple]:
        """Returns a list of events (same types as StreamingReasoningDetector)."""
        events = []
        reasoning = delta.get(self._field_name)
        content = delta.get("content")

        if reasoning:
            self._reasoning_parts.append(reasoning)
            self._pending_step_text += reasoning

            # Check step boundaries
            for pattern in STEP_BOUNDARY_PATTERNS:
                match = pattern.search(self._pending_step_text)
                if match and match.start() >= STEP_MIN_CHARS:
                    step_text = self._pending_step_text[:match.start()].strip()
                    remaining = self._pending_step_text[match.end():]
                    if len(step_text) >= STEP_MIN_CHARS:
                        step = ReasoningStep(index=self._step_index, text=step_text)
                        self._step_index += 1
                        events.append(("reasoning_step", step))
                        self._pending_step_text = remaining
                        return events

            if len(self._pending_step_text) >= STEP_MAX_CHARS:
                step = ReasoningStep(
                    index=self._step_index,
                    text=self._pending_step_text.strip(),
                )
                self._step_index += 1
                events.append(("reasoning_step", step))
                self._pending_step_text = ""

            return events

        if content:
            if self._pending_step_text.strip() and len(self._pending_step_text.strip()) >= STEP_MIN_CHARS:
                step = ReasoningStep(
                    index=self._step_index,
                    text=self._pending_step_text.strip(),
                    is_final=True,
                )
                self._step_index += 1
                events.append(("reasoning_step", step))
                events.append(("reasoning_ended",))
                self._pending_step_text = ""

            self._output_parts.append(content)
            events.append(("output", content))
            return events

        return [("empty", "")]

    def finalize(self) -> DetectionResult:
        return DetectionResult(
            reasoning_text="".join(self._reasoning_parts),
            output_tokens=self._output_parts,
            raw_text="".join(self._reasoning_parts) + "".join(self._output_parts),
        )

    @property
    def has_reasoning(self) -> bool:
        return bool(self._reasoning_parts)

    @property
    def pending_step_text(self) -> str:
        return self._pending_step_text

    @property
    def step_count(self) -> int:
        return self._step_index
