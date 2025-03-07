import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
#from dotenv import load_dotenv
import os

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm_registry import LLMRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import async_timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as VLLM
from vllm import SamplingParams
import torch
import os
import time
import json
from AgentPrune.llm.llm import LLM

model_cache = {}

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


def worker_task(rank):
    # 子进程的逻辑
    torch.cuda.set_device(rank)
    print(f"Process {rank} is using CUDA device {torch.cuda.current_device()}.")

async def achat(
    model_str: str,
    messages: List[Dict],
) -> Union[List[str], str]:
    model_str=model_str[5:]
    global model_cache
    if model_str not in model_cache:
        print(f'Loading model: {model_str}')
        if model_str=='qwen-0.5b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Qwen2.5-0.5B-Instruct"
        if model_str=='qwen-3b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Qwen2.5-3B-Instruct"
        if model_str=='qwen-7b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Qwen2.5-7B-Instruct"
        if model_str=='qwen-14b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Qwen2.5-14B-Instruct"
        if model_str=='qwen-32b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/models--Qwen--Qwen2.5-32B-Instruct"
        if model_str=='qwen-72b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
        if model_str=='qwen-7b-sft':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/LLaMA-Factory/model/qwen-7b"
        if model_str=='qwen-7b-mix':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/LLaMA-Factory/model/qwen-7b-mix"
        if model_str=='qwen-7b-math':
            model_name="/cpfs01/shared/ma4agi/tangshengji/agent_hub/Qwen/Qwen2.5-Math-7B-Instruct"
        if model_str=='qwen-14b-deepseek':
            model_name='/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B'
        if model_str=='qwen-7b-deepseek':
            model_name='/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B'
        if model_str=='qwen-1.5b-deepseek':
            model_name='/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B'
        if model_str=='qwen-32b-deepseek':
            model_name='/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/DeepSeek-R1-Distill-Qwen-32B'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        llm = VLLM(model=model_name,gpu_memory_utilization=0.9)
        model_cache[model_str] = (model, tokenizer)
    else:
        model, tokenizer = model_cache[model_str]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    outputs = llm.generate(text,use_tqdm=True)
    print(outputs)
    return outputs

@LLMRegistry.register('VllmChat')
class VllmChat(LLM):

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
        
        
