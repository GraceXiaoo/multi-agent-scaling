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
base_url = "https://boyuerichdata.chatgptten.com/v1/chat/completions"

skey = "sk-VnDc44GxxRmzrsY61O6zjCpzNfg0czlg7um4CDewIY5Vye5x"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {skey}"  
}

def calculate_token_count(model: str, messages: List[Dict]) -> int:
    """
    计算输入或输出消息的 token 数量。
    :param model: 使用的模型名称（如 gpt-4、gpt-3.5-turbo 等）
    :param messages: 消息列表（包含 role 和 content）
    :return: 消息的总 token 数量
    """
    # 根据模型类型选择适当的编码
    if "gpt-4" in model:
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif "gpt-3.5" in model:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        encoding = tiktoken.encoding_for_model("gpt-4o")

    # 计算每条消息的 token 数量
    num_tokens = 0
    for message in messages:
        # 每条消息的格式是 {"role": "xxx", "content": "xxx"}
        num_tokens += 4  # 每条消息的开销：role + content + 开始/结束标记
        num_tokens += len(encoding.encode(message["role"]))  # 计算 role 的 token 数量
        num_tokens += len(encoding.encode(message["content"]))  # 计算 content 的 token 数量
    
    num_tokens += 2  # 对话结束的额外标记
    return num_tokens


import requests
async def achat(
    model_str: str,
    messages: List[Dict],
) -> Union[List[str], str]:
    model_name=''
    if model_str=='gpt-4o-mini':
        model_name="gpt-4o-mini-2024-07-18"
    elif model_str=='gpt-4o':
        model_name="gpt-4o-2024-11-20"
    data = {"model": model_name, # 可以替换为老师需要的模型
    "messages": messages,
    "temperature": 0.5}
    time.sleep(0.2)
    response = requests.post(base_url, headers=headers, json=data)

    # 计算输入 token 数量
    input_token_count = calculate_token_count(model_str, messages)
    print(f"Input token count: {input_token_count}")
    
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        output_token_count = calculate_token_count(model_str, [{"role": "assistant", "content": content}])
        print(f"Output token count: {output_token_count}")

        print("Response JSON:",content)
        return (content,input_token_count,output_token_count)
    else:
        max_try=0
        while response.status_code != 200 and max_try<5:
            max_try+=1
            time.sleep(1)
            print('failed sleeping count：{max_try}')
            response = requests.post(base_url, headers=headers, json=data)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                output_token_count = calculate_token_count(model_str, [{"role": "assistant", "content": content}])
                print(f"Output token count: {output_token_count}")

                print("Response JSON:",content)
                return (content,input_token_count,output_token_count)
        return (response.text,input_token_count,0)

from AgentPrune.llm.llm import LLM
@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

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
        