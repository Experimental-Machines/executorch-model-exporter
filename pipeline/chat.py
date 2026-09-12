"""Smoke-test prompts, rendered the way the app feeds the runtime.

The 1.4.0 runtime never adds BOS, so the app writes the family's BOS as text and relies on
the tokenizer to encode it (openweights PromptTemplate.kt). Rendering the upstream chat
template with ``bos_token`` filled in produces the same kind of prompt.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

QUESTION = "What is the capital of France? Answer with one word."
COMPLETION = "The capital of France is"
EXPECTED = "paris"


def _template(model_dir: Path, tokenizer_config: dict) -> str | None:
    standalone = model_dir / "chat_template.jinja"
    if standalone.exists():
        return standalone.read_text(encoding="utf-8")
    template = tokenizer_config.get("chat_template")
    if isinstance(template, list):  # named templates: use the default one
        by_name = {t.get("name"): t.get("template") for t in template}
        template = by_name.get("default") or next(iter(by_name.values()), None)
    return template


def _token_text(value) -> str:
    if isinstance(value, dict):
        return value.get("content", "")
    return value or ""


def _environment():
    # The environment transformers renders chat templates in.
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def render(model_dir: Path, tokenizer_config: dict, instruct: bool) -> str:
    """The prompt string for the smoke test."""
    template = _template(model_dir, tokenizer_config) if instruct else None
    bos = _token_text(tokenizer_config.get("bos_token"))
    if template is None:
        return (bos if tokenizer_config.get("add_bos_token") else "") + COMPLETION
    variables = {
        "messages": [{"role": "user", "content": QUESTION}],
        "add_generation_prompt": True,
        "bos_token": bos,
        "eos_token": _token_text(tokenizer_config.get("eos_token")),
        "enable_thinking": False,
    }
    return _environment().from_string(template).render(**variables)
