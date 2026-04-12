"""
Wrapper for Stable Diffusion 1.5 image generation.
Provides a simple interface for generating images from text prompts on Apple MPS / CUDA / CPU.
"""

import gc
import torch
from pathlib import Path

try:
    from diffusers import StableDiffusionPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("Warning: diffusers not found. Install with: pip install diffusers accelerate")


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class StableDiffusionGenerator:
    """Wrapper for Stable Diffusion 1.5 image generation."""

    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None):
        if not HAS_DIFFUSERS:
            raise ImportError(
                "diffusers library not found. Install with: pip install diffusers accelerate"
            )

        if device is None:
            device = _get_device()

        self.device = device
        self.model_id = model_id

        print(f"Loading Stable Diffusion model: {model_id}...")
        print(f"Device: {device}")

        dtype = torch.float16 if device != "cpu" else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        print("Stable Diffusion model loaded")

    def generate(
        self,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = None,
    ) -> str:
        """
        Generate an image from a text prompt and save it.

        Args:
            prompt: English text prompt for image generation.
            output_path: Path to save the generated image (.png).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Image width in pixels.
            height: Image height in pixels.
            seed: Random seed for reproducibility.

        Returns:
            Path to the saved image.
        """
        generator = None
        if seed is not None:
            # MPS requires generator on CPU
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        # Free MPS cache after each generation
        if self.device == "mps":
            torch.mps.empty_cache()

        return output_path

    def cleanup(self):
        """Release model memory."""
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe = self.pipe.to("cpu")
            del self.pipe
            self.pipe = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


# --------------- module-level lazy singleton ---------------

_sd_instance = None


def generate_image(prompt: str, output_path: str, **kwargs) -> str:
    """Generate a single image (lazy-loads the model on first call)."""
    global _sd_instance
    if _sd_instance is None:
        _sd_instance = StableDiffusionGenerator()
    return _sd_instance.generate(prompt, output_path, **kwargs)


def cleanup_sd():
    """Clean up the Stable Diffusion singleton and free memory."""
    global _sd_instance
    if _sd_instance is not None:
        _sd_instance.cleanup()
        _sd_instance = None
    gc.collect()


if __name__ == "__main__":
    if not HAS_DIFFUSERS:
        print("diffusers is not installed. Please install it first:")
        print("  pip install diffusers accelerate")
        exit(1)

    print("Testing Stable Diffusion wrapper...")
    try:
        gen = StableDiffusionGenerator()
        out = gen.generate(
            prompt="A watercolor painting of a quiet Russian village in winter, snow-covered rooftops",
            output_path="/tmp/sd_test.png",
            seed=42,
        )
        print(f"Image saved to {out}")
        gen.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
