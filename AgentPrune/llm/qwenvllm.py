from vllm import LLM as VLLM
from vllm import SamplingParams
prompts = ["""<|im_start|>user 
    hello<|im_end|>
    <|im_start|>assistant
    """,
    'hello']
#llm = LLM(model="/cpfs01/user/xiaojin/xiaojin/hf_models/Mixtral-8x7B-Instruct-v0.1")

model="/cpfs01/user/xiaojin/xiaojin/hf_models/Qwen2.5-0.5B-Instruct"
model="/cpfs01/user/xiaojin/xiaojin/hf_models/Mixtral-8x7B-Instruct-v0.1"
model='/cpfs01/user/xiaojin/xiaojin/hf_models/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct'
model='/cpfs01/user/xiaojin/xiaojin/hf_models/DeepSeek-V2-Lite-Chat'
llm = VLLM(model=model,gpu_memory_utilization=0.9,tensor_parallel_size=1,trust_remote_code=True)

outputs = llm.generate(prompts,use_tqdm=True)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
