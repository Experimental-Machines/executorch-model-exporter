"""Load an exported .pte the way the phone does and check it produces sane text.

``TextLLMRunner`` is the C++ runner the app's ``LlmModule`` wraps, shipped in the Linux
executorch wheel. The metadata methods are read back with ``executorch.runtime`` so the
published config.json reports what the file says, not what the export was asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import chat

_METADATA_PREFIXES = ("get_", "use_", "enable_")
# Below this size a model may not know the answer; only degenerate output fails it.
ANSWER_REQUIRED_FROM_PARAMS = 500_000_000


def _plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def read_metadata(pte: Path) -> dict:
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(pte)
    metadata = {}
    for name in sorted(program.method_names):
        if not name.startswith(_METADATA_PREFIXES):
            continue
        outputs = [_plain(v) for v in program.load_method(name).execute([])]
        if name == "get_eos_ids":
            flat = []
            for value in outputs:
                flat.extend(value if isinstance(value, list) else [value])
            metadata[name] = [int(v) for v in flat]
        else:
            value = outputs[0] if len(outputs) == 1 else outputs
            metadata[name] = value
    metadata["methods"] = sorted(program.method_names)
    return metadata


def generate(pte: Path, tokenizer: Path, prompt: str, max_new_tokens: int) -> tuple[list[str], dict]:
    # The wheel's runner links only portable and XNNPACK kernels. The exported graph also
    # calls llama::custom_sdpa / update_cache and quantized_decomposed::embedding_byte,
    # whose kernels register into portable_lib's operator registry when these libraries
    # load, in this order (examples/models/llama/runner/native.py). The app's AAR links
    # them statically.
    # isort: off
    from executorch.extension.pybindings import portable_lib  # noqa: F401
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.kernels import quantized  # noqa: F401
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    # isort: on

    runner = TextLLMRunner(str(pte), str(tokenizer))
    pieces: list[str] = []
    stats: dict = {}
    config = GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=0.0, num_bos=0, num_eos=0)
    runner.generate(
        prompt,
        config,
        token_callback=pieces.append,
        stats_callback=lambda s: stats.update(json.loads(s.to_json_string())),
    )
    return pieces, stats


def degenerate(pieces: list[str]) -> bool:
    """Eight or more tokens that are all the same piece."""
    return len(pieces) >= 8 and len(set(pieces)) == 1


def run(
    pte: Path,
    tokenizer: Path,
    model_dir: Path,
    tokenizer_config: dict,
    instruct: bool,
    total_params: int,
    max_new_tokens: int = 32,
) -> dict:
    prompt = chat.render(model_dir, tokenizer_config, instruct)
    problems = []
    try:
        pieces, stats = generate(pte, tokenizer, prompt, max_new_tokens)
    except RuntimeError as error:  # the runner reports load/generation failures this way
        pieces, stats = [], {}
        problems.append(f"runner error: {error}")
    text = "".join(pieces)
    answered = chat.EXPECTED in text.lower()
    required = total_params >= ANSWER_REQUIRED_FROM_PARAMS
    if not pieces and not problems:
        problems.append("generated no tokens")
    if degenerate(pieces):
        problems.append(f"degenerate output: {pieces[0]!r} repeated")
    if required and not answered:
        problems.append(f"expected {chat.EXPECTED!r} in the reply")
    return {
        "prompt": prompt,
        "reply": text,
        "tokens": len(pieces),
        "answered": answered,
        "answer_required": required,
        "passed": not problems,
        "problems": problems,
        "stats": stats,
    }
