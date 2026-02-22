"""
Reasoning token detector.

Consumes streamed deltas from the upstream API and classifies each token
as either *reasoning* or *output*.  Uses model-specific profiles from config.

Design decision: We buffer the full stream and classify post-hoc rather than
inline, because some models emit the </think> tag split across multiple chunks.
This avoids misclassification at token boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from gateway.config import ReasoningProfile


@dataclass
class DetectionResult:
    """Final output of the reasoning detector after the stream ends."""
    reasoning_text: str       # Extracted reasoning content (markers stripped)
    output_tokens: list[str]  # Ordered list of output token strings
    raw_text: str             # Full concatenated stream text


class StreamingReasoningDetector:
    """
    Incrementally fed tokens; emits output tokens in real-time and
    accumulates reasoning tokens for later summarisation.

    States:
        INIT       → haven't seen any reasoning marker yet
        REASONING  → inside a reasoning block
        OUTPUT     → past the reasoning block; forwarding output tokens

    For models using `delta_field` (e.g., reasoning_content), the caller
    should route tokens directly; this detector handles the content-pattern
    approach.
    """

    def __init__(self, profile: ReasoningProfile) -> None:
        self._profile = profile
        self._buffer = ""  # accumulated raw text
        self._reasoning_parts: list[str] = []
        self._output_parts: list[str] = []
        self._state: str = "INIT"  # INIT | REASONING | OUTPUT
        self._reasoning_ended = False

    # -- Public API ----------------------------------------------------------

    def feed(self, token: str) -> list[tuple[str, str]]:
        """
        Feed a new token.  Returns a list of (classification, text) pairs
        that should be acted upon immediately.

        classification is one of: "reasoning", "output", "buffered"
        "buffered" means the token is held internally (boundary ambiguity).
        """
        self._buffer += token
        events: list[tuple[str, str]] = []

        if self._state == "OUTPUT":
            # Already past reasoning – everything is output
            self._output_parts.append(token)
            events.append(("output", token))
            return events

        if self._state == "INIT":
            # Check if reasoning has started
            if self._profile.start_pattern.search(self._buffer):
                self._state = "REASONING"
                # Everything before the match is output preamble (rare)
                match = self._profile.start_pattern.search(self._buffer)
                if match and match.start() > 0:
                    pre = self._buffer[:match.start()]
                    self._output_parts.append(pre)
                    events.append(("output", pre))
                # Don't emit reasoning tokens yet – buffer them
                return events
            else:
                # No reasoning detected yet.  If we've accumulated a lot of
                # text without seeing a start marker, assume no reasoning.
                if len(self._buffer) > 200:
                    self._state = "OUTPUT"
                    self._output_parts.append(self._buffer)
                    events.append(("output", self._buffer))
                else:
                    events.append(("buffered", token))
                return events

        if self._state == "REASONING":
            # Check if reasoning has ended
            if self._profile.end_pattern.search(self._buffer):
                self._state = "OUTPUT"
                self._reasoning_ended = True

                match = self._profile.end_pattern.search(self._buffer)
                start_match = self._profile.start_pattern.search(self._buffer)

                if start_match and match:
                    # Extract reasoning content between markers
                    r_start = start_match.end() if self._profile.strip_markers else start_match.start()
                    r_end = match.start() if self._profile.strip_markers else match.end()
                    reasoning = self._buffer[r_start:r_end]
                    self._reasoning_parts = [reasoning]

                    # Anything after end marker is output
                    after = self._buffer[match.end():]
                    if after:
                        self._output_parts.append(after)
                        events.append(("output", after))
                    events.insert(0, ("reasoning_complete", reasoning))
                return events
            else:
                events.append(("buffered", token))
                return events

        return events

    def finalize(self) -> DetectionResult:
        """
        Call when the upstream stream ends.  Flushes any buffered content.
        """
        # If we never found an end marker, treat everything as output
        if self._state == "REASONING":
            # Reasoning never closed – likely no real reasoning
            reasoning = ""
            output_text = self._buffer
            # Strip start marker if present
            match = self._profile.start_pattern.search(self._buffer)
            if match:
                output_text = self._buffer[:match.start()] + self._buffer[match.end():]
            self._output_parts = [output_text]
        elif self._state == "INIT":
            # Never saw reasoning at all
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


class DeltaFieldDetector:
    """
    For models that emit reasoning in a separate delta field
    (e.g., `reasoning_content`).  Much simpler – no pattern matching.
    """

    def __init__(self, field_name: str = "reasoning_content") -> None:
        self._field_name = field_name
        self._reasoning_parts: list[str] = []
        self._output_parts: list[str] = []

    def classify_delta(self, delta: dict) -> tuple[str, str]:
        """Returns (classification, text)."""
        reasoning = delta.get(self._field_name)
        content = delta.get("content")
        if reasoning:
            self._reasoning_parts.append(reasoning)
            return ("reasoning", reasoning)
        if content:
            self._output_parts.append(content)
            return ("output", content)
        return ("empty", "")

    def finalize(self) -> DetectionResult:
        return DetectionResult(
            reasoning_text="".join(self._reasoning_parts),
            output_tokens=self._output_parts,
            raw_text="".join(self._reasoning_parts) + "".join(self._output_parts),
        )

    @property
    def has_reasoning(self) -> bool:
        return bool(self._reasoning_parts)
