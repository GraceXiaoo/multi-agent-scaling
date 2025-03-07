from typing import Optional
from class_registry import ClassRegistry

from AgentPrune.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o"
        if model_name == 'mock':
            model = cls.registry.get(model_name)
        if model_name.startswith('qwen'):
            model = cls.registry.get('QwenChat',model_name)
        # if model_name.startswith('qwen'):
        #     model = cls.registry.get('QwenChat1',model_name)
        if model_name.startswith('llama'):
            model = cls.registry.get('LlamaChat',model_name)
        if model_name.startswith('moe'):
            model = cls.registry.get('MoeChat',model_name)
        if model_name.startswith('mamba'):
            model = cls.registry.get('MambaChat',model_name)
        if model_name.startswith('vllm'):
            model = cls.registry.get('VllmChat',model_name)
        if model_name.startswith('gpt'):
            model = cls.registry.get('GPTChat',model_name)
        if model_name.startswith('test'):
            model = cls.registry.get('TestChat',model_name)
        if model_name.startswith('deepseek'):
            model = cls.registry.get('DeepseekChat',model_name)
        return model
