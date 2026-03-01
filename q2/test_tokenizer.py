"""
Validates the tokenizer_config.json fix without needing model weights or vocab files.
Checks:
  1. chat_template field exists and is non-empty
  2. Template contains expected Qwen3 structural markers
  3. README base_model is correct
"""
import json

print("=== Testing tokenizer_config.json ===\n")

with open("q2/broken-model/tokenizer_config.json") as f:
    config = json.load(f)

# 1. chat_template must exist
assert "chat_template" in config, "FAIL: chat_template field is missing"
assert config["chat_template"], "FAIL: chat_template is empty"
print("[PASS] chat_template field exists")

# 2. Must use Qwen's im_start/im_end tokens
tmpl = config["chat_template"]
assert "<|im_start|>" in tmpl, "FAIL: missing <|im_start|> in template"
assert "<|im_end|>" in tmpl, "FAIL: missing <|im_end|> in template"
print("[PASS] Template uses <|im_start|> / <|im_end|> tokens")

# 3. Must handle enable_thinking flag
assert "enable_thinking" in tmpl, "FAIL: enable_thinking not handled in template"
print("[PASS] Template handles enable_thinking flag")

# 4. Must handle think tags
assert "<think>" in tmpl and "</think>" in tmpl, "FAIL: think tags missing"
print("[PASS] Template contains <think> / </think> markers")

# 5. tokenizer_class should be Qwen2Tokenizer
assert config["tokenizer_class"] == "Qwen2Tokenizer", f"FAIL: unexpected tokenizer_class: {config['tokenizer_class']}"
print("[PASS] tokenizer_class = Qwen2Tokenizer")

# 6. eos_token should be <|im_end|>
assert config["eos_token"] == "<|im_end|>", f"FAIL: eos_token = {config['eos_token']}"
print("[PASS] eos_token = <|im_end|>")

print("\n=== Testing README.md ===\n")

with open("q2/broken-model/README.md") as f:
    readme = f.read()

assert "Qwen/Qwen3-8B" in readme, "FAIL: README does not reference Qwen/Qwen3-8B"
assert "meta-llama/Meta-Llama-3.1-8B" not in readme.split("---")[0], \
    "FAIL: Wrong base_model still in README front-matter"
print("[PASS] README base_model correctly set to Qwen/Qwen3-8B")

print("\nAll checks passed.")
