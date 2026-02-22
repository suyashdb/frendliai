#!/usr/bin/env python3
"""
Client script – demonstrates calling the FriendliAI Reasoning Gateway.

Features:
  - Streams SSE response with phase-aware rendering
  - Color-coded output for each phase (prompt summary, reasoning summary, output)
  - Supports any OpenAI-compatible gateway endpoint

Usage:
    python client.py
    python client.py --gateway http://localhost:8000 --model deepseek-r1
    python client.py --prompt "Explain quantum entanglement"
    python client.py --no-prompt-summary --no-reasoning-summary
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="FriendliAI Gateway Client")
    p.add_argument("--gateway", default="http://localhost:8000", help="Gateway base URL")
    p.add_argument("--model", default="deepseek-r1", help="Model name to request")
    p.add_argument("--prompt", default="What is the capital of France?", help="User prompt")
    p.add_argument("--system", default=None, help="Optional system message")
    p.add_argument("--no-prompt-summary", action="store_true", help="Disable prompt summary")
    p.add_argument("--no-reasoning-summary", action="store_true", help="Disable reasoning summary")
    p.add_argument("--warm", action="store_true", help="Pre-warm a session before sending the request")
    return p.parse_args()


PHASE_STYLES = {
    "prompt_summary": {"color": "cyan", "title": "📋 Prompt Summary", "border": "cyan"},
    "reasoning_summary": {"color": "yellow", "title": "🧠 Reasoning", "border": "yellow"},
    "output": {"color": "green", "title": "✅ Response", "border": "green"},
}


def _phase_title(phase: str, label: str, step: int | None = None) -> str:
    """Build display title for a phase, including step number if present."""
    style = PHASE_STYLES.get(phase, {})
    base_title = style.get("title", label)
    if step is not None:
        return f"{base_title} (Step {step})"
    return base_title


def warm_session(
    gateway_url: str,
    model: str,
    system: str | None = None,
    prompt: str | None = None,
) -> str | None:
    """Pre-warm a session and return the session_id."""
    url = f"{gateway_url.rstrip('/')}/v1/sessions/warm"

    body: dict = {"model": model}
    if system:
        body["system_prompt"] = system
    if prompt:
        body["messages"] = [{"role": "user", "content": prompt}]

    console.print("[dim]🔥 Pre-warming session...[/dim]")
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body)
        if resp.status_code != 200:
            console.print(f"[red]Warmup failed: HTTP {resp.status_code}[/red]")
            return None
        data = resp.json()
        session_id = data["session_id"]
        console.print(f"[dim]   Session: {session_id}[/dim]")

    # Wait a moment for background warmup tasks to complete
    import time
    time.sleep(1.5)

    # Check session status
    with httpx.Client(timeout=10.0) as client:
        status_resp = client.get(f"{gateway_url.rstrip('/')}/v1/sessions/{session_id}")
        if status_resp.status_code == 200:
            status = status_resp.json()
            summary_ready = status.get("prompt_summary_ready", False)
            kv_warm = status.get("kv_cache_warmed", False)
            console.print(
                f"[dim]   Summary cached: {'✅' if summary_ready else '⏳'} | "
                f"KV warm: {'✅' if kv_warm else '⏳'}[/dim]"
            )

    return session_id


def stream_gateway(
    gateway_url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    include_prompt_summary: bool = True,
    include_reasoning_summary: bool = True,
    session_id: str | None = None,
):
    """Stream from the gateway and render phase-aware output."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "include_prompt_summary": include_prompt_summary,
        "include_reasoning_summary": include_reasoning_summary,
    }
    if session_id:
        body["session_id"] = session_id

    url = f"{gateway_url.rstrip('/')}/v1/chat/completions"

    current_phase = None
    current_phase_key = None  # unique key per phase instance
    phase_contents: dict[str, str] = {}
    phase_labels: dict[str, str] = {}

    console.print()
    console.rule("[bold blue]FriendliAI Reasoning Gateway[/bold blue]")
    warm_tag = " | [bold green]🔥 WARM SESSION[/bold green]" if session_id else ""
    console.print(f"[dim]Model: {model} | Gateway: {gateway_url}[/dim]{warm_tag}")
    console.print(f"[dim]Prompt: {prompt}[/dim]")
    console.print()

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                console.print(f"[red]Error: HTTP {resp.status_code}[/red]")
                console.print(resp.read().decode())
                return

            for line in resp.iter_lines():
                if not line:
                    continue

                # Phase change event
                if line.startswith("event: phase"):
                    continue  # next line has the data

                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # Check if this is a phase event
                    if "phase" in data and "label" in data:
                        # Flush previous phase
                        if current_phase_key and current_phase_key in phase_contents:
                            _render_phase(
                                current_phase,
                                phase_contents[current_phase_key],
                                phase_labels.get(current_phase_key, ""),
                            )

                        phase = data["phase"]
                        step = data.get("step")
                        label = data.get("label", "")
                        # Create unique key (reasoning_summary can appear multiple times)
                        phase_key = f"{phase}_{step}" if step else phase
                        current_phase = phase
                        current_phase_key = phase_key
                        phase_contents[phase_key] = ""
                        title = _phase_title(phase, label, step)
                        phase_labels[phase_key] = title
                        continue

                    # Regular content chunk
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content and current_phase_key:
                        phase_contents[current_phase_key] = (
                            phase_contents.get(current_phase_key, "") + content
                        )
                        style = PHASE_STYLES.get(current_phase, {})
                        console.print(content, end="", style=style.get("color", "white"))

    # Flush final phase
    console.print()  # newline after streaming
    if current_phase_key and current_phase_key in phase_contents:
        _render_phase(
            current_phase,
            phase_contents[current_phase_key],
            phase_labels.get(current_phase_key, ""),
        )

    console.print()
    console.rule("[bold blue]Done[/bold blue]")


def _render_phase(phase: str, content: str, title: str = ""):
    """Render a completed phase as a panel."""
    style = PHASE_STYLES.get(phase, {"title": phase, "border": "white"})
    display_title = title or style.get("title", phase)
    console.print()
    console.print(
        Panel(
            content.strip(),
            title=display_title,
            border_style=style["border"],
            padding=(0, 1),
        )
    )


def main():
    args = parse_args()
    try:
        session_id = None
        if args.warm:
            session_id = warm_session(
                gateway_url=args.gateway,
                model=args.model,
                system=args.system,
                prompt=args.prompt,
            )

        stream_gateway(
            gateway_url=args.gateway,
            model=args.model,
            prompt=args.prompt,
            system=args.system,
            include_prompt_summary=not args.no_prompt_summary,
            include_reasoning_summary=not args.no_reasoning_summary,
            session_id=session_id,
        )
    except httpx.ConnectError:
        console.print(
            "[red]Could not connect to gateway. Is the server running?[/red]\n"
            f"[dim]Tried: {args.gateway}[/dim]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted[/dim]")


if __name__ == "__main__":
    main()
