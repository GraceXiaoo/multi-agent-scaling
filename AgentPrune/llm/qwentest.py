from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/DeepSeek-R1-Distill-Qwen-32B"

#3b没弄好

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
hello
"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
model.generation_config.pad_token_id = tokenizer.pad_token_id
print()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
    pad_token_id=tokenizer.eos_token_id
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)