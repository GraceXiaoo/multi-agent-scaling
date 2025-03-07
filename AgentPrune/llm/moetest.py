from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb

bnb.nn.Linear4bit.bnb_4bit_compute_dtype = torch.float16
model_id = "/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Falcon3-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_flash_attention_2=True,device_map="auto")

text = "Hello my name is"

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

tokens = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
            )


# 计算输入 token 长度
input_token_length = tokens['input_ids'].shape[1]  # 输入序列的长度
print('#' * 10)
print("Input Token Length:", input_token_length)
