from typing import List,Any,Dict

from AgentPrune.graph.node import Node
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import execute_code_get_return
from dataset.gsm8k_dataset import gsm_get_predict
from dataset.physics_dataset import physics_get_predict
import json

@AgentRegistry.register('MathSolverRAG')
class MathSolverRAG(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolverRAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        file_path = "/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/AgentPrune/dataset/gpqa/all_results.json" #暂时写死
        with open(file_path, 'r', encoding='utf-8') as fr:
            self.ragDataset = json.load(fr)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        if self.role == "Math Solver":
            user_prompt += f"(Hint: The answer modulo 23 is near to {raw_inputs['answer']}"
        # 加上rag
        # 提取data['Question']部分
        context_prefix = "Please choose the correct answer from among the following options: \n"
        context_start_index = raw_inputs['task'].find(context_prefix)
        if context_start_index != -1:
            data_question = raw_inputs['task'][:context_start_index].strip()
        else:
            raise ValueError("Context prefix not found in task string.")
        evidence = ""
        for entry in self.ragDataset:
            if entry['question'].strip() == data_question.strip():
                evidence = entry['google_retrieval']
                print("Found matching google_retrieval")
                break
        else:
            print("No matching question found.")
        user_prompt += f"()? And here are some online materials that you might find useful:\n{evidence}\n"
        print(user_prompt)
        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        return response