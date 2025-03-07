from typing import List,Any,Dict

from AgentPrune.graph.node import Node
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import execute_code_get_return
from dataset.gsm8k_dataset import gsm_get_predict
from dataset.physics_dataset import physics_get_predict
import json
import os
import re
from typing import Optional, Tuple, List, Dict
import time

from AgentPrune.agents.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from AgentPrune.agents.evaluate import (
    extract_answer
)
from AgentPrune.agents.prompts import(
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
MAX_SEARCH_LIMIT=1

jina_api_key='jina_cc14825c926845f9b2300365c7615f08GR0nK9nPdesTWUExcfER4qV2fizi'
bing_subscription_key='7606ef9a3d30744c9dcb3ca505b54a66cdb052f9908cf71b057f426b8ba742d6'


@AgentRegistry.register('Search_O1_RAG')
class Search_O1_RAG(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "Search_O1_RAG" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """ 
        question=raw_inputs["task"]
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=question,role=self.role)
        system_prompt = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
        model_path=raw_inputs["llm"]
        if 'deepseek' in model_path.lower():
            user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
        elif 'llama' in model_path.lower():
            user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
        else:
            user_prompt = get_task_instruction_multi_choice(question)
        return system_prompt, user_prompt

    def get_cache(self,inputs):
        dataset_name=inputs['dataset']
        cache_dir = f'/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/AgentPrune/result/cache/{dataset_name}/'
        search_cache_path = os.path.join(cache_dir, 'search_cache.json')
        url_cache_path = os.path.join(cache_dir, 'url_cache.json')
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(search_cache_path):
            with open(search_cache_path, 'r', encoding='utf-8') as f:
                search_cache = json.load(f)
        else:
            search_cache = {}

        if os.path.exists(url_cache_path):
            with open(url_cache_path, 'r', encoding='utf-8') as f:
                url_cache = json.load(f)
        else:
            url_cache = {}
        return search_cache,url_cache

    
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    def generate_webpage_to_reasonchain_batch(
        self,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict],  # New parameter to collect outputs21
        max_tokens: int = 32768,
        coherent: bool = False,
    ) -> List[str]:
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]
        prompts = [{"role": "user", "content": up} for up in user_prompts]

        outputs=[]
        for prompt in prompts:
            output = self.llm.gen(prompt)
            outputs.append(output)
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in outputs]

        for i, (p, r, e) in enumerate(zip(prompts, outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e
            })
        return extracted_infos
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    def extract_between(self,text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
    def refine_search(self,raw_input,output,prompt,**kwargs):
        MAX_SEARCH_LIMIT=1
        MAX_TURN=1
        top_k=10
        max_doc_len=3000

        search_cache,url_cache=self.get_cache(raw_input)
        seq= {
        'item': raw_input,#不确定是啥
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
        }
        turn = 0
        while True:
            if not seq['finished']:
                turn+=1
                print(f'\n-------------- Turn {turn} --------------')
                # Initialize batch variables
                batch_relevant_info = []
                batch_original_questions = []
                batch_prev_reasonings = []
                batch_search_queries = []
                batch_documents = []
                batch_sequences = []
                # Collect URLs to fetch across all sequences
                all_urls_to_fetch = set()
                url_snippets = {}
                url_sequence_map = {}  # Map URL to list of sequences needing it
                text =output[0]
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text
                # Extract search query
                search_query = self.extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                print('*'*30)
                print(search_query)
                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                results = bing_web_search(search_query)
                                search_cache[search_query] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = {}
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:top_k]
                        seq['relevant_info'] = relevant_info
                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")
                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['task'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)
                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")
                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=True,
                        jina_api_key=jina_api_key,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content
            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents)
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=dataset_name,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    max_tokens=max_tokens,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis in zip(batch_sequences, webpage_analyses):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = self.replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

            # Check if all sequences are finished
            if seq['finished']:
                break
            else:
                if turn >= MAX_TURN:
                    print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                    break
        return seq

    def replace_recent_steps(self,origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps


    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'user','content':system_prompt+user_prompt}]
        print('#'*20)
        print('message:',message)
        #print(message)
        response = await self.llm.agen(message)
        print('#'*20)
        print('response',response)
        seq=self.refine_search(input,response,message)
        print('#'*20)
        print('seq',seq)
        #得到第一次的结果
        return seq['output'],response[1],response[2]