import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
#from dotenv import load_dotenv
import os

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm_registry import LLMRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI
import asyncio
import async_timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
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
        if model_str=='moe-7b-dense':
            model_name="/nas/shared/ma4agi/model/Mistral-7B-Instruct-v0.3"
        if model_str=='moe-7b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Mixtral-8x7B-Instruct-v0.1"
        if model_str=='moe-22b':
            model_name="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/hf_models/Mixtral-8x22B-Instruct-v0.1"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 确保 pad_token_id 被设置
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model_cache[model_str] = (model, tokenizer)
    else:
        model, tokenizer = model_cache[model_str]
    if model_str=='moe-23b':
        text = messages[0]['content']+'\n'+messages[1]['content']
        inputs = tokenizer(text, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,pad_token_id=tokenizer.pad_token_id, max_new_tokens=1024)
        response_content=tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response_content
    else:
        try:
            messages=[{'role':'user','content':messages[0]['content']+'\n'+messages[1]['content']}]
            tokens = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
            )
            # 将 BatchEncoding 中的每个张量移动到 GPU，并转换为 float16
            tokens = tokens.to("cuda")  # 仅支持移动设备
            #tokens["input_ids"] = tokens["input_ids"].to(dtype=torch.float16)
            #tokens["attention_mask"] = tokens["attention_mask"].to(dtype=torch.float16)
            generated_ids = model.generate(**tokens, max_new_tokens=1024, do_sample=True)

            # decode with HF tokenizer
            response_content = tokenizer.decode(generated_ids[0])
            print('#'*10)


            #新增加的
            # 计算输入 token 长度
            input_token_length = tokens['input_ids'].shape[1]  # 输入序列的长度
            # 计算输出 token 长度
            output_token_length = generated_ids.shape[1]  # 生成序列的长度
            print("Input Token Length:", input_token_length)
            print("Output Token Length:", output_token_length)



            return (response_content,input_token_length,output_token_length)

        except asyncio.TimeoutError:
            print('Timeout')
            raise TimeoutError("Moe Timeout")

from AgentPrune.llm.llm import LLM
@LLMRegistry.register('MoeChat')
class MoeChat(LLM):

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
        