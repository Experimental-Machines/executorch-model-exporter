from pipeline import chat

QWEN3_LIKE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
LLAMA_LIKE = (
    "{{ bos_token }}{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
    "{% endfor %}"
)


def test_instruct_prompt_renders_upstream_template_without_thinking(tmp_path):
    prompt = chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=True)
    assert prompt.startswith("<|im_start|>user\nWhat is the capital of France?")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_bos_is_written_as_text(tmp_path):
    prompt = chat.render(tmp_path, {"chat_template": LLAMA_LIKE, "bos_token": "<|begin_of_text|>"}, instruct=True)
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>user")


def test_standalone_template_file_wins(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("X{{ messages[0].content }}", encoding="utf-8")
    assert chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=True) == "X" + chat.QUESTION


def test_base_models_get_a_completion_prompt(tmp_path):
    assert chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=False) == chat.COMPLETION
    with_bos = {"bos_token": {"content": "<s>"}, "add_bos_token": True}
    assert chat.render(tmp_path, with_bos, instruct=False) == "<s>" + chat.COMPLETION
