"""
Streamlit web interface for image sequence generation from long-reads.
"""

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from web_extractor import fetch_and_extract, is_url
from contextlib import redirect_stdout, redirect_stderr
from urllib.parse import urlparse

st.set_page_config(page_title="Image Sequence Generation", layout="wide")

# ── session state ────────────────────────────────────────────────────

if "processing" not in st.session_state:
    st.session_state.processing = False
if "results" not in st.session_state:
    st.session_state.results = None

st.title("Image Sequence Generation from Long-reads")
st.markdown("Upload a Russian text file or enter a URL to generate a set of images.")

# ── sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    num_blocks = st.slider("Number of blocks", 3, 15, 7, help="How many images to generate")
    num_steps = st.slider("Inference steps", 20, 50, 30, help="More steps = better quality, slower")
    guidance_scale = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)
    seed = st.number_input("Seed", value=42, step=1, help="For reproducibility")

    st.markdown("---")
    st.header("Model Info")
    st.markdown("""
    **Segmentation:** Qwen 2.5 1.5B
    **Image gen:** Stable Diffusion 1.5
    **Language:** Russian input, English prompts
    """)

# ── input ────────────────────────────────────────────────────────────

st.header("Input")
input_method = st.radio("Choose input method:", ["URL", "Text File"], horizontal=True)

input_data = None
input_type = None
filename = None

if input_method == "URL":
    url_input = st.text_input("Enter URL:", placeholder="https://dzen.ru/a/...")
    if url_input:
        if is_url(url_input):
            input_data = url_input
            input_type = "url"
            parsed = urlparse(url_input)
            domain = parsed.netloc.replace("www.", "").split(".")[0]
            path_parts = [p for p in parsed.path.split("/") if p]
            name = path_parts[-1][:50] if path_parts else "index"
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            filename = f"web_{domain}_{name}" if name else f"web_{domain}"
        else:
            st.error("Please enter a valid URL (must start with http:// or https://)")
else:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        input_data = uploaded_file.read().decode("utf-8")
        input_type = "file"
        filename = Path(uploaded_file.name).stem

# ── run button ───────────────────────────────────────────────────────

run_button = st.button(
    "Generate Images",
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
        st.info("Processing... First run may take a few minutes to download models.")
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
        status_text.text("Step 1/2: Extracting text...")
        progress_bar.progress(5)

        start_parse = time.time()
        if input_type == "url":
            with redirect_stdout(log_capture), redirect_stderr(log_capture):
                raw = fetch_and_extract(input_data, save_to=None)
        else:
            raw = input_data
            log_capture.write(f"Reading from uploaded file\n")

        results["time_parsing"] = time.time() - start_parse
        results["input_chars"] = len(raw)
        results["input_words"] = len(raw.split())
        log_capture.write(
            f"Input: {results['input_chars']:,} chars, {results['input_words']:,} words\n"
        )

        # Step 2: Segmentation + Generation
        status_text.text("Step 2/2: Segmenting text and generating prompts...")
        progress_bar.progress(10)

        output_dir = Path("data/images") / results["filename"]

        def update_progress(current, total):
            # segmentation is ~10%, generation fills 10%-95%
            pct = 10 + int(85 * current / total)
            progress_bar.progress(min(pct, 95))
            status_text.text(f"Generating image {current}/{total}...")

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

    st.header("Results")

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images", results.get("num_blocks", len(blocks)))
        col2.metric("Total time", f"{results['time_total']:.1f}s")
        col3.metric("Per image", f"{results.get('time_per_image_avg', 0):.1f}s")
        col4.metric("Input words", f"{results['input_words']:,}")

        st.markdown("**Time Breakdown:**")
        if results["time_parsing"] > 0:
            st.write(f"- Parsing: {results['time_parsing']:.2f}s")
        st.write(f"- Segmentation + prompts: {results.get('time_segmentation', 0):.1f}s")
        st.write(f"- Image generation: {results.get('time_generation', 0):.1f}s")
        st.write(f"- **Total: {results['time_total']:.1f}s**")

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
