import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Придумай сказку на русском длиной 500 слов"
    messages = [
        {"role": "system", "content": "Ты Квен. Ты полезный помощник"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == '__main__':
    main()
