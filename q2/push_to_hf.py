"""
Pushes the broken-model to suyashdb/broken-model-fixed in two commits:
  1. Original broken files (as-is from yunmorning/broken-model)
  2. Fixed files with the chat_template added and README corrected
"""
from huggingface_hub import HfApi
import os

REPO_ID = "suyashdb/broken-model-fixed"
api = HfApi()

# ── Commit 1: original broken files ──────────────────────────────────────────
print("Uploading original (broken) files as initial commit...")

ORIGINAL_README = """\
---
library_name: transformers
pipeline_tag: text-generation
base_model:
- meta-llama/Meta-Llama-3.1-8B
---
"""

ORIGINAL_TOKENIZER_CONFIG = open(
    os.path.join(os.path.dirname(__file__), "broken-model/tokenizer_config.json")
).read()

# The original tokenizer_config had no chat_template — strip it out to recreate original state
import json
original_tc = json.loads(ORIGINAL_TOKENIZER_CONFIG)
original_tc.pop("chat_template", None)  # remove the fix to restore original broken state
import io

api.upload_file(
    path_or_fileobj=ORIGINAL_README.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Initial commit: original broken model files (from yunmorning/broken-model)",
)
print("  [OK] README.md")

api.upload_file(
    path_or_fileobj=json.dumps(original_tc, indent=2).encode(),
    path_in_repo="tokenizer_config.json",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Initial commit: original broken model files (from yunmorning/broken-model)",
)
print("  [OK] tokenizer_config.json (original, no chat_template)")

# Upload the unchanged files directly from disk
for filename in ["config.json", "generation_config.json", "merges.txt", ".gitattributes"]:
    filepath = os.path.join(os.path.dirname(__file__), "broken-model", filename)
    if os.path.exists(filepath):
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Initial commit: original broken model files (from yunmorning/broken-model)",
        )
        print(f"  [OK] {filename}")

print("\nCommit 1 done.\n")

# ── Commit 2: fixed files ─────────────────────────────────────────────────────
print("Uploading fixed files...")

fixed_readme_path = os.path.join(os.path.dirname(__file__), "broken-model/README.md")
fixed_tc_path = os.path.join(os.path.dirname(__file__), "broken-model/tokenizer_config.json")

api.upload_file(
    path_or_fileobj=fixed_readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="fix: add missing chat_template, correct base_model to Qwen/Qwen3-8B",
)
print("  [OK] README.md (fixed)")

api.upload_file(
    path_or_fileobj=fixed_tc_path,
    path_in_repo="tokenizer_config.json",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="fix: add missing chat_template, correct base_model to Qwen/Qwen3-8B",
)
print("  [OK] tokenizer_config.json (with chat_template added)")

print("\nCommit 2 done.")
print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")
