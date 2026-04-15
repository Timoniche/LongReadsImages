"""
Segment long Russian text into semantic blocks and generate images for each block.
Uses Qwen 2.5 1.5B for text segmentation + prompt generation,
and Stable Diffusion 1.5 for image generation.
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from image_generator import generate_image, cleanup_sd

# ── device ────────────────────────────────────────────────────────────

def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _get_device()

# ── Qwen model (loaded once at module level) ─────────────────────────

SUM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading Qwen model: {SUM_MODEL} ...")

# MPS cannot handle tensors > 4 GB; Qwen 1.5B triggers this during generation.
# Run the text model on CPU (fast enough) and reserve MPS for Stable Diffusion.
_qwen_device = "cpu" if DEVICE == "mps" else DEVICE
_qwen_dtype = torch.float16 if _qwen_device != "cpu" else torch.float32
print(f"Device: {_qwen_device} (GPU reserved for image generation)")

tok = AutoTokenizer.from_pretrained(SUM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    SUM_MODEL,
    dtype=_qwen_dtype,
    device_map=_qwen_device,
)
model.eval()
_loaded = True
print("Qwen model loaded")

# ── context extraction (pass 1) ─────────────────────────────────────

_CONTEXT_SYSTEM_PROMPT = """\
You are an assistant that extracts visual context from Russian texts.
You will receive a long Russian text. Analyze it and respond ONLY with a JSON object:

{
  "style": "a single visual style for ALL illustrations, e.g. digital illustration, watercolor, editorial art",
  "palette": "color palette description, e.g. warm earth tones, muted blues and greys",
  "entities": [
    {"name": "entity name", "description": "stable English visual description for Stable Diffusion, 10-15 words"},
    ...
  ]
}

Rules:
- Extract up to 5 most important recurring entities (characters, locations, key objects).
- The "description" must be a fixed English visual anchor that will be repeated in every image prompt where this entity appears. Include appearance details: clothing, colors, age, distinguishing features.
- Choose ONE consistent art style and color palette for the entire series.
- Respond with JSON only, no extra text."""


def _extract_global_context(text: str, max_input_tokens: int = 4000) -> dict | None:
    """
    Pass 1: extract key entities, visual style, and color palette from the text.
    Returns dict with 'style', 'palette', 'entities' or None on failure.
    """
    tokens = tok.encode(text, add_special_tokens=False)
    truncated = tok.decode(tokens[:max_input_tokens], skip_special_tokens=True) if len(tokens) > max_input_tokens else text
    del tokens

    messages = [
        {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Извлеки визуальный контекст из следующего русского текста:\n\n{truncated}"},
    ]

    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("  Pass 1: extracting global context (entities, style) ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tok.eos_token_id or tok.pad_token_id, use_cache=True,
        )
    elapsed = time.time() - t0
    print(f"  Context extracted in {elapsed:.1f}s")

    raw = tok.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # parse JSON
    try:
        ctx = json.loads(raw)
        if "style" in ctx and "entities" in ctx:
            print(f"  Style: {ctx['style']}")
            print(f"  Entities: {[e['name'] for e in ctx.get('entities', [])]}")
            return ctx
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            ctx = json.loads(m.group())
            if "style" in ctx and "entities" in ctx:
                print(f"  Style: {ctx['style']}")
                print(f"  Entities: {[e['name'] for e in ctx.get('entities', [])]}")
                return ctx
        except json.JSONDecodeError:
            pass

    print("  Warning: failed to extract context, proceeding without anchoring")
    return None


# ── segmentation (pass 2) ───────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an assistant that analyzes Russian texts and creates image descriptions.
You will receive a Russian text and must:
1. Split it into {num_blocks} semantic blocks.
2. For each block, write a short English image prompt suitable for Stable Diffusion.

IMPORTANT CONSISTENCY RULES:
- ALL prompts must use this art style: {style}
- ALL prompts must use this color palette: {palette}
- When any of the following entities appear in a block, use their EXACT description:
{entities_section}
- End every prompt with: "{style}, {palette}"

Respond ONLY with a JSON array. Each element must have:
- "block_text": the first 120 characters of the original Russian text for this block
- "prompt": an English image generation prompt (2-3 sentences, descriptive, visual)

Example:
[
  {{"block_text": "Первые 120 символов блока...", "prompt": "A detailed painting of ..."}},
  {{"block_text": "Следующий фрагмент...", "prompt": "A photograph of ..."}}
]
"""

_SYSTEM_PROMPT_NO_CONTEXT = """\
You are an assistant that analyzes Russian texts and creates image descriptions.
You will receive a Russian text and must:
1. Split it into {num_blocks} semantic blocks.
2. For each block, write a short English image prompt suitable for Stable Diffusion.

Respond ONLY with a JSON array. Each element must have:
- "block_text": the first 120 characters of the original Russian text for this block
- "prompt": an English image generation prompt (2-3 sentences, descriptive, visual)

Example:
[
  {{"block_text": "Первые 120 символов блока...", "prompt": "A detailed painting of ..."}},
  {{"block_text": "Следующий фрагмент...", "prompt": "A photograph of ..."}}
]
"""


def segment_and_prompt(text: str, num_blocks: int = 7, max_input_tokens: int = 8000,
                       global_context: dict | None = None) -> list[dict]:
    """
    Split Russian *text* into *num_blocks* semantic blocks and generate an
    English image prompt for each block.

    If *global_context* is provided (from _extract_global_context), prompts
    will include consistent style/entity anchors.

    Returns a list of dicts: {"block_text": str, "prompt": str}
    """
    # truncate if too long
    tokens = tok.encode(text, add_special_tokens=False)
    if len(tokens) > max_input_tokens:
        print(f"  Input too long ({len(tokens)} tokens), truncating to {max_input_tokens}")
        text = tok.decode(tokens[:max_input_tokens], skip_special_tokens=True)
    del tokens

    if global_context:
        entities_lines = "\n".join(
            f"  - {e['name']}: {e['description']}"
            for e in global_context.get("entities", [])
        ) or "  (no entities extracted)"
        system_msg = _SYSTEM_PROMPT.format(
            num_blocks=num_blocks,
            style=global_context.get("style", "digital illustration"),
            palette=global_context.get("palette", "natural colors"),
            entities_section=entities_lines,
        )
    else:
        system_msg = _SYSTEM_PROMPT_NO_CONTEXT.format(num_blocks=num_blocks)

    user_msg = (
        f"Разбей следующий русский текст на {num_blocks} смысловых блоков "
        f"и для каждого блока напиши описание картинки на английском языке "
        f"для генерации через Stable Diffusion.\n\nТекст:\n{text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("  Generating segmentation + prompts ...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tok.eos_token_id or tok.pad_token_id,
            use_cache=True,
        )
    elapsed = time.time() - t0
    new_tokens = len(outputs[0]) - inputs["input_ids"].shape[-1]
    print(f"  Done in {elapsed:.1f}s ({new_tokens} tokens generated)")

    raw_output = tok.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # ── parse JSON ────────────────────────────────────────────────
    blocks = _parse_blocks_json(raw_output)
    if blocks is not None:
        _fill_full_text(blocks, text)
        return blocks

    # ── fallback: split by paragraphs ─────────────────────────────
    print("  JSON parse failed, falling back to paragraph splitting")
    return _fallback_paragraph_split(text, num_blocks)


def _fill_full_text(blocks: list[dict], original_text: str) -> None:
    """
    Restore full text for each block by matching block_text anchors
    against the original text. Each block gets a 'full_text' key.
    """
    # find positions of each anchor in the original text
    positions = []
    for block in blocks:
        anchor = block.get("block_text", "")[:80]
        if not anchor:
            positions.append(-1)
            continue
        pos = original_text.find(anchor)
        if pos == -1:
            # fuzzy: try first 40 chars
            pos = original_text.find(anchor[:40])
        positions.append(pos)

    for i, block in enumerate(blocks):
        start = positions[i]
        if start == -1:
            # couldn't locate -- use block_text as-is
            block["full_text"] = block.get("block_text", "")
            continue

        # end = start of next block, or end of text
        end = len(original_text)
        for j in range(i + 1, len(blocks)):
            if positions[j] != -1:
                end = positions[j]
                break

        block["full_text"] = original_text[start:end].strip()


def _parse_blocks_json(raw: str) -> list[dict] | None:
    """Try to extract a JSON array from the model output."""
    # try direct parse
    try:
        arr = json.loads(raw)
        if isinstance(arr, list) and all("prompt" in b for b in arr):
            return arr
    except json.JSONDecodeError:
        pass

    # try regex extraction
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and all("prompt" in b for b in arr):
                return arr
        except json.JSONDecodeError:
            pass

    return None


def _fallback_paragraph_split(text: str, num_blocks: int) -> list[dict]:
    """Split text by paragraphs and build basic prompts via a per-block LLM call."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # merge paragraphs into roughly equal chunks
    chunk_size = max(1, len(paragraphs) // num_blocks)
    chunks: list[str] = []
    for i in range(0, len(paragraphs), chunk_size):
        chunks.append("\n".join(paragraphs[i : i + chunk_size]))
    chunks = chunks[:num_blocks]

    blocks = []
    for idx, chunk in enumerate(chunks):
        prompt = _single_block_prompt(chunk)
        blocks.append({
            "block_text": chunk[:120],
            "full_text": chunk,
            "prompt": prompt,
        })
        print(f"  Fallback block {idx + 1}/{len(chunks)} done")

    return blocks


def _single_block_prompt(block_text: str) -> str:
    """Generate an English SD prompt for a single Russian text block."""
    messages = [
        {
            "role": "user",
            "content": (
                "Write a short English image generation prompt (2-3 sentences) "
                "suitable for Stable Diffusion that illustrates the following Russian text:\n\n"
                f"{block_text[:500]}"
            ),
        }
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150, do_sample=False,
            pad_token_id=tok.eos_token_id or tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


# ── image generation ──────────────────────────────────────────────────

def generate_images(
        blocks: list[dict],
        output_dir: Path,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        progress_callback=None,
) -> list[dict]:
    """
    Generate one image per block and enrich each dict with 'image_path' and
    'generation_time' keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, block in enumerate(blocks):
        t0 = time.time()
        out_path = str(output_dir / f"block_{i}.png")
        generate_image(
            prompt=block["prompt"],
            output_path=out_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + i if seed is not None else None,
        )
        block["image_path"] = out_path
        block["generation_time"] = time.time() - t0
        block["block_index"] = i

        if progress_callback:
            progress_callback(i + 1, len(blocks))

        print(f"  Image {i + 1}/{len(blocks)} saved ({block['generation_time']:.1f}s)")

    return blocks


# ── full pipeline ─────────────────────────────────────────────────────

def run_pipeline(
        text: str,
        output_dir: str | Path,
        num_blocks: int = 7,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        progress_callback=None,
) -> dict:
    """
    End-to-end pipeline: segment text → generate prompts → generate images.

    Returns a results dict with timing info, blocks and metadata.
    """
    output_dir = Path(output_dir)
    t_total_start = time.time()

    # Step 1a: extract global context (entities, style, palette)
    t0 = time.time()
    global_context = _extract_global_context(text)
    time_context = time.time() - t0

    # Step 1b: segmentation + prompt generation (with context anchors)
    t0 = time.time()
    blocks = segment_and_prompt(text, num_blocks=num_blocks, global_context=global_context)
    time_segmentation = time.time() - t0 + time_context

    # Step 2: image generation
    t0 = time.time()
    blocks = generate_images(
        blocks,
        output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        progress_callback=progress_callback,
    )
    time_generation = time.time() - t0

    time_total = time.time() - t_total_start

    return {
        "input_chars": len(text),
        "input_words": len(text.split()),
        "num_blocks": len(blocks),
        "time_segmentation": round(time_segmentation, 2),
        "time_generation": round(time_generation, 2),
        "time_total": round(time_total, 2),
        "time_per_image_avg": round(time_generation / max(len(blocks), 1), 2),
        "blocks": blocks,
        "output_dir": str(output_dir),
        "global_context": global_context,
    }


# ── cleanup ───────────────────────────────────────────────────────────

def cleanup_models():
    """Release Qwen and SD models and free memory."""
    global model, tok, _loaded

    # Qwen
    if "model" in globals() and model is not None:
        try:
            model.cpu()
            del model
        except Exception as e:
            print(f"Warning cleaning up Qwen: {e}")

    if "tok" in globals() and tok is not None:
        try:
            del tok
        except Exception:
            pass

    # SD
    cleanup_sd()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    _loaded = False
    print("Models cleaned up")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if input_arg is None:
        filename = "example_1"
        path = f"data/longreads/{filename}.txt"
        raw = Path(path).read_text(encoding="utf-8")
        print(f"Reading from file: {path}")
    else:
        path = input_arg
        filename = Path(path).stem
        raw = Path(path).read_text(encoding="utf-8")
        print(f"Reading from file: {path}")

    print(f"\nInput text: {len(raw):,} characters, {len(raw.split()):,} words\n")

    out_dir = Path("data/images") / filename
    results = run_pipeline(raw, out_dir)

    print(f"\nDone! {results['num_blocks']} images saved to {results['output_dir']}")
    print(f"  Segmentation: {results['time_segmentation']:.1f}s")
    print(f"  Generation:   {results['time_generation']:.1f}s  ({results['time_per_image_avg']:.1f}s/image)")
    print(f"  Total:        {results['time_total']:.1f}s")
