"""
Streamlit web interface for image sequence generation from long-reads.
"""

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from contextlib import redirect_stdout, redirect_stderr

# Optional CLIP scoring
try:
    from sentence_transformers import SentenceTransformer
    from PIL import Image
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

st.set_page_config(page_title="Image Sequence Generation", layout="wide")


@st.cache_resource
def load_clip_models():
    """Load CLIP image + text models once and cache across reruns."""
    if not HAS_CLIP:
        return None, None
    img_model = SentenceTransformer("clip-ViT-B-32")
    txt_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
    return img_model, txt_model


def calculate_clip_score(image_path: str, text: str, img_model, txt_model) -> float:
    """Cosine similarity between image and text via CLIP."""
    if img_model is None or txt_model is None:
        return 0.0
    img = Image.open(image_path)
    img_emb = img_model.encode(img)
    txt_emb = txt_model.encode(text)
    cos_sim = float(
        np.dot(img_emb, txt_emb)
        / (np.linalg.norm(img_emb) * np.linalg.norm(txt_emb) + 1e-10)
    )
    return round(cos_sim, 4)

# ── session state ────────────────────────────────────────────────────

if "processing" not in st.session_state:
    st.session_state.processing = False
if "results" not in st.session_state:
    st.session_state.results = None

st.title("Генерация последовательности изображений по лонг-ридам")
st.markdown("Загрузите статью на русском в формате txt для генерации изображений")

# ── defaults for removed sliders ────────────────────────────────────
num_blocks = 7
num_steps = 30
guidance_scale = 7.5

# ── input ────────────────────────────────────────────────────────────

st.header("Входные данные")

input_data = None
input_type = None
filename = None

uploaded_file = st.file_uploader("Загрузите текстовый файл", type=["txt"])
if uploaded_file is not None:
    input_data = uploaded_file.read().decode("utf-8")
    input_type = "file"
    filename = Path(uploaded_file.name).stem

# ── settings (fixed defaults, not shown in UI) ───────────────────────

seed = 42
compute_clip = HAS_CLIP

# ── run button ───────────────────────────────────────────────────────

run_button = st.button(
    "Сгенерировать изображения",
    type="primary",
    disabled=(input_data is None or st.session_state.processing),
)

# ── processing ───────────────────────────────────────────────────────

if run_button and input_data is not None:
    st.session_state.processing = True
    st.session_state.results = None

    log_capture = io.StringIO()
    progress_container = st.container()

    with progress_container:
        st.info("Выполняю... Первый запуск может занять несколько минут для загрузки моделей.")
        progress_bar = st.progress(0)
        status_text = st.empty()

    results = {
        "filename": filename or "output",
        "input_type": input_type,
        "input_chars": 0,
        "input_words": 0,
        "time_parsing": 0.0,
        "time_segmentation": 0.0,
        "time_generation": 0.0,
        "time_total": 0.0,
        "blocks": [],
        "logs": "",
    }

    total_start = time.time()

    try:
        from pipeline import run_pipeline

        # Step 1: Extract text
        status_text.text("Шаг 1/2: Извлекаю текст...")
        progress_bar.progress(5)

        start_parse = time.time()
        raw = input_data
        log_capture.write(f"Reading from uploaded file\n")

        results["time_parsing"] = time.time() - start_parse
        results["input_chars"] = len(raw)
        results["input_words"] = len(raw.split())
        log_capture.write(
            f"Input: {results['input_chars']:,} chars, {results['input_words']:,} words\n"
        )

        # Step 2: Segmentation + Generation
        status_text.text("Шаг 2/2: Сегментирую текст и составляю промпты...")
        progress_bar.progress(10)

        output_dir = Path("data/images") / results["filename"]

        def update_progress(current, total):
            # segmentation is ~10%, generation fills 10%-95%
            pct = 10 + int(85 * current / total)
            progress_bar.progress(min(pct, 95))
            status_text.text(f"Генерирую изображение {current}/{total}...")

        with redirect_stdout(log_capture), redirect_stderr(log_capture):
            pipeline_results = run_pipeline(
                text=raw,
                output_dir=output_dir,
                num_blocks=num_blocks,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=update_progress,
            )

        results["blocks"] = pipeline_results["blocks"]
        results["time_segmentation"] = pipeline_results["time_segmentation"]
        results["time_generation"] = pipeline_results["time_generation"]
        results["time_total"] = time.time() - total_start
        results["time_per_image_avg"] = pipeline_results["time_per_image_avg"]
        results["num_blocks"] = pipeline_results["num_blocks"]
        results["global_context"] = pipeline_results.get("global_context")

        # CLIP scoring
        if compute_clip and HAS_CLIP:
            status_text.text("Вычисляю CLIP метрику (гружу модели для первого запуска)...")
            clip_img_model, clip_txt_model = load_clip_models()
            clip_scores = []
            for idx, block in enumerate(results["blocks"]):
                img_path = block.get("image_path", "")
                block_text = block.get("block_text", "")
                if img_path and Path(img_path).exists() and block_text:
                    score = calculate_clip_score(img_path, block_text, clip_img_model, clip_txt_model)
                    block["clip_score"] = score
                    clip_scores.append(score)
                    status_text.text(f"CLIP scoring {idx + 1}/{len(results['blocks'])}...")
                else:
                    block["clip_score"] = None
            results["clip_scores"] = clip_scores
            results["clip_score_avg"] = round(float(np.mean(clip_scores)), 4) if clip_scores else 0.0

        progress_bar.progress(100)
        status_text.text("Done!")
        time.sleep(0.5)

        results["logs"] = log_capture.getvalue()
        st.session_state.results = results
        st.session_state.processing = False
        st.rerun()

    except Exception as e:
        import traceback
        log_capture.write(f"\nError: {e}\n")
        log_capture.write(traceback.format_exc())
        results["logs"] = log_capture.getvalue()
        st.session_state.results = results
        st.session_state.processing = False
        st.error(f"Error: {e}")
        st.rerun()

# ── display results ──────────────────────────────────────────────────

if st.session_state.results is not None and not st.session_state.processing:
    results = st.session_state.results

    st.header("Результат генерации")

    blocks = results.get("blocks", [])

    if not blocks:
        st.warning("No images were generated.")
    else:
        # ── Article view: text with inline images ────────────────
        st.subheader("Illustrated Article")

        for i, block in enumerate(blocks):
            # Text of this block
            full_text = block.get("full_text") or block.get("block_text", "")
            # Render paragraphs
            for paragraph in full_text.split("\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    st.markdown(paragraph)

            # Image after the text block
            img_path = block.get("image_path", "")
            if img_path and Path(img_path).exists():
                st.image(img_path, use_container_width=True)
            else:
                st.error(f"Image {i + 1} not found")

            st.markdown("---")

        # ── Statistics ───────────────────────────────────────────
        st.subheader("Statistics")

        clip_avg = results.get("clip_score_avg")
        has_clip_scores = clip_avg is not None

        n_cols = 5 if has_clip_scores else 4
        cols = st.columns(n_cols)
        cols[0].metric("Images", results.get("num_blocks", len(blocks)))
        cols[1].metric("Total time", f"{results['time_total']:.1f}s")
        cols[2].metric("Per image", f"{results.get('time_per_image_avg', 0):.1f}s")
        cols[3].metric("Input words", f"{results['input_words']:,}")
        if has_clip_scores:
            cols[4].metric("CLIP avg", f"{clip_avg:.4f}")

        st.markdown("**Time Breakdown:**")
        if results["time_parsing"] > 0:
            st.write(f"- Parsing: {results['time_parsing']:.2f}s")
        st.write(f"- Segmentation + prompts: {results.get('time_segmentation', 0):.1f}s")
        st.write(f"- Image generation: {results.get('time_generation', 0):.1f}s")
        st.write(f"- **Total: {results['time_total']:.1f}s**")

        # ── Per-block quality table ─────────────────────────────
        if has_clip_scores:
            import pandas as pd
            st.markdown("**Per-block CLIP Scores:**")
            table_rows = []
            for i, block in enumerate(blocks):
                score = block.get("clip_score")
                text_preview = (block.get("block_text") or "")[:80]
                if len(block.get("block_text") or "") > 80:
                    text_preview += "..."
                prompt_preview = (block.get("prompt") or "")[:80]
                if len(block.get("prompt") or "") > 80:
                    prompt_preview += "..."
                table_rows.append({
                    "Block": i + 1,
                    "CLIP Score": f"{score:.4f}" if score is not None else "N/A",
                    "Gen time (s)": f"{block.get('generation_time', 0):.1f}",
                    "Text": text_preview,
                    "Prompt": prompt_preview,
                })
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ZIP download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for block in blocks:
                p = block.get("image_path", "")
                if p and Path(p).exists():
                    zf.write(p, f"block_{block.get('block_index', 0)}.png")
        zip_buffer.seek(0)
        st.download_button(
            "Download all images (ZIP)",
            data=zip_buffer,
            file_name=f"{results['filename']}_images.zip",
            mime="application/zip",
        )

    # Global context (prompt anchoring)
    ctx = results.get("global_context")
    if ctx:
        with st.expander("Visual Context (Prompt Anchoring)", expanded=False):
            st.write(f"**Style:** {ctx.get('style', 'N/A')}")
            st.write(f"**Palette:** {ctx.get('palette', 'N/A')}")
            entities = ctx.get("entities", [])
            if entities:
                st.write("**Entities:**")
                for e in entities:
                    st.write(f"- **{e['name']}**: {e['description']}")

    # Prompts (technical details)
    with st.expander("Generated Prompts", expanded=False):
        for i, block in enumerate(blocks):
            st.markdown(f"**Block {i + 1}:** _{block.get('prompt', 'N/A')}_")
            gen_time = block.get("generation_time", 0)
            if gen_time:
                st.caption(f"Generation time: {gen_time:.1f}s")

    # Logs
    with st.expander("Processing Logs", expanded=False):
        st.code(results.get("logs", ""), language=None)
        st.download_button(
            label="Download Logs",
            data=results.get("logs", ""),
            file_name=f"{results['filename']}_logs.txt",
            mime="text/plain",
        )
