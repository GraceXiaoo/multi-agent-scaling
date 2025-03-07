import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
#from dotenv import load_dotenv
import os
import traceback
from AgentPrune.llm.format import Message
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import async_timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json
from typing import Tuple
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def achat(
    model_str: str,
    messages: List[Dict],
) -> Tuple[str, int, int]:
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

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map="auto",
            use_flash_attention_2=False
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

        # print('input text')
        # print(text)
        #print(text)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        if 'deepseek' in model_str.lower():
            max_tokens = 20480
        else:
            max_tokens = 8192
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response_content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #计算token数量
        input_token_length = model_inputs['input_ids'].shape[1]
        # print(f'input token length: {input_token_length}')
        # print(input_token_length)
        # print('out token length')
        output_token_length = generated_ids[0].shape[0]
        # print(f"out token length: {output_token_length}")
        output = (response_content,input_token_length,output_token_length)
        if type(output) is not tuple:
            print('不是tuple')
            print(input_token_length) 
            print(output_token_length)
        #print(response_content)
        # print('output:', output)
        # print('!!!!!!!!!!!!!!!!!')
        return output
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("Qwen Timeout")
    except Exception as e:
        # 捕获所有异常并打印详细的错误信息
        print(f"An error occurred: {e}")
        print("Stack trace:")
        traceback.print_exc()
        raise  # 重新抛出异常，保持程序崩溃或进行后续处理

from AgentPrune.llm.llm import LLM
@LLMRegistry.register('QwenChat')
class QwenChat(LLM):

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
        
        
