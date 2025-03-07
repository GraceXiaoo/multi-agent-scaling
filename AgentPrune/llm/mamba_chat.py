import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
#from dotenv import load_dotenv
import os

from AgentPrune.llm.format import Message

from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI
import asyncio
import async_timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json

model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def achat(
    model_str: str,
    messages: List[Dict],
) -> Union[List[str], str]:
    global model_cache
    if model_str not in model_cache:
        print(f'Loading model: {model_str}')
        if model_str=='mamba-1b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Falcon3-1B-Instruct"
        if model_str=='mamba-7b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Falcon3-Mamba-7B-Instruct"
        if model_str=='mamba-40b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/falcon-40b-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        # Cache the model and tokenizer
        model_cache[model_str] = (model, tokenizer)
    else:
        model, tokenizer = model_cache[model_str]

    try:
        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response_content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #计算token数量
        input_token_length = model_inputs['input_ids'].shape[1]
        print('input token length')
        print(input_token_length)
        print('out token length')
        output_token_length = generated_ids[0].shape[0]
        print(f"输出 Token 长度: {output_token_length}")
        #print(response_content)
        return (response_content,input_token_length,output_token_length)

    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("Qwen Timeout")

from AgentPrune.llm.llm import LLM
@LLMRegistry.register('MambaChat')
class MambaChat(LLM):

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
        