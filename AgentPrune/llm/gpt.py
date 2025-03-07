import os
from openai import OpenAI
import time 

# 记录开始时间
start_time = time.time()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-5a5dd944e6794717b486a601360a3493",  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=[
        {'role': 'user', 'content': '9.9和9.11谁大'}
    ]
)
print(completion)
# 记录结束时间
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.6f} 秒")
# 通过reasoning_content字段打印思考过程
print("思考过程：")
print(completion.choices[0].message.reasoning_content)
print("token数量")
print(completion.usage.prompt_tokens,completion.usage.completion_tokens)

# 通过content字段打印最终答案
print("最终答案：")
print(completion.choices[0].message.content)