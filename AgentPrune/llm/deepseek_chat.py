import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
#from dotenv import load_dotenv
import os
from openai import OpenAI
from AgentPrune.llm.format import Message

from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI
import asyncio
import async_timeout
import os
import time
import json
import tiktoken

'''

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model,
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            response_data = await response.json()
            if isinstance(response_data['data'],str):
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt,response_data['data'],model)
                print(response['data'])
                return response_data['data']
            else:
                raise Exception("api error")
'''


'''
@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    messages: List[Dict],
) -> Union[List[str], str]:

    api_kwargs: Dict[str, Any]
    api_kwargs = dict(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            response = await aclient.chat.completions.create(model=model, messages=messages)
            time.sleep(3)
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("GPT Timeout")
    response_content = response.choices[0].message.content

    if isinstance(response_content,str):
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response_content, model)
        return response_content

'''
start_time = time.time()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-5a5dd944e6794717b486a601360a3493",  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


import requests
async def achat(
    model_str: str,
    messages: List[Dict],
) -> Union[List[str], str]:
    completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=messages
    )
    # 计算输入 token 数量
    input_token_count = completion.usage.prompt_tokens
    print(f"Input token count: {input_token_count}")
    
    #思考过程
    print(completion)
    content=completion.choices[0].message.reasoning_content
    #最终答案
    ans=completion.choices[0].message.reasoning_content
    #计算输出tokens
    output_token_count = completion.usage.completion_tokens
    print(f"Output token count: {output_token_count}")

    print("推理过程:",content)
    print("最终结果:",ans)
    return (ans,input_token_count,output_token_count)

from AgentPrune.llm.llm import LLM
@LLMRegistry.register('DeepseekChat')
class DeepseekChat(LLM):

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
        