from abc import ABC, abstractmethod
from typing import List, Union, Optional

from AgentPrune.llm.format import Message


class LLM(ABC):
    def __init__(self) -> None:
        self.DEFAULT_MAX_TOKENS = 30000
        
        self.DEFAULT_TEMPERATURE = 0.7
        self.DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass
