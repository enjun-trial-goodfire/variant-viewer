"""Batch-generate descriptions for heads missing them in builds/heads.json.

Calls Claude API to generate a 1-sentence description for each head based on
its key, display name, group, and category. Writes back to the same file.

Usage:
    ANTHROPIC_API_KEY=... python3 pipeline/generate_descriptions.py [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

HEADS_PATH = Path("builds/heads.json")

SYSTEM = """\
You are a genomics annotation expert. Given a probe head's key, display name, \
group, and category, write a single concise sentence (max 25 words) explaining \
what this annotation detects or measures. Be specific and technical. Do not \
start with "This head" or "Detects". Use active voice.

Examples:
- "Predicted probability that the position overlaps a CDS exon"
- "Alpha helix secondary structure prediction from AlphaFold2"
- "H3K27ac active enhancer mark in GM12878 lymphoblastoid cells"
- "Whether the amino acid substitution changes the residue's net charge"
"""


def build_batch(heads: dict) -> list[dict]:
    """Build a list of (key, prompt) for heads missing descriptions."""
    batch = []
    for key, info in sorted(heads.items()):
        if info.get("description"):
            continue
        display = info.get("display", key)
        group = info.get("group", "Unknown")
        category = info.get("category", "unknown")
        prompt = (
            f"Head key: {key}\n"
            f"Display name: {display}\n"
            f"Group: {group}\n"
            f"Category: {category}\n"
            f"\nWrite one sentence describing what this annotation measures."
        )
        batch.append({"key": key, "prompt": prompt})
    return batch


def generate(client: anthropic.Anthropic, batch: list[dict], heads: dict) -> int:
    """Generate descriptions for all items in batch. Returns count of successes."""
    count = 0
    total = len(batch)
    for i, item in enumerate(batch):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                temperature=0.3,
                system=SYSTEM,
                messages=[{"role": "user", "content": item["prompt"]}],
            )
            text = resp.content[0].text.strip().rstrip(".")
            # Remove quotes if the model wraps in them
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            heads[item["key"]]["description"] = text
            count += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}] {item['key']}: {text[:60]}...")
        except Exception as e:
            print(f"  ERROR {item['key']}: {e}", file=sys.stderr)
            time.sleep(1)
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate head descriptions with Claude")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without calling API")
    parser.add_argument("--heads", type=Path, default=HEADS_PATH, help="Path to heads.json")
    args = parser.parse_args()

    with open(args.heads) as f:
        data = json.load(f)
    heads = data.get("heads", {})

    batch = build_batch(heads)
    print(f"Found {len(batch)} heads missing descriptions (out of {len(heads)} total)")

    if args.dry_run:
        for item in batch[:10]:
            print(f"  {item['key']}")
        if len(batch) > 10:
            print(f"  ... and {len(batch) - 10} more")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    print(f"Generating descriptions for {len(batch)} heads...")
    count = generate(client, batch, heads)
    print(f"Generated {count}/{len(batch)} descriptions")

    with open(args.heads, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote updated {args.heads}")


if __name__ == "__main__":
    main()
