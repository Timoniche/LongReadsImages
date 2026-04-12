from diffusers import StableDiffusionPipeline
import torch

def main():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("mps")

    prompt = "cute labrador on the green grass"
    image = pipe(prompt).images[0]

    image.save("cute_labrador.png")

if __name__ == '__main__':
    main()