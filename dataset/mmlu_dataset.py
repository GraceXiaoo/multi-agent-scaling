import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:
        self._split = split
        data_path = f"/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/AgentPrune/dataset/mmlu/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        # List all the JSONL files in the directory (if there are multiple)
        jsonl_paths = glob.glob(data_path + "*.jsonl")
        jsonl_paths = sorted(jsonl_paths)
        print("Number of files: ", len(jsonl_paths))

        # Initialize the dataframe with expected column names
        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in jsonl_paths:
            # # Read the JSONL file, assume the JSON is structured with the same keys
            # single_df = pd.read_json(path, lines=True)
            
            # # Ensure the dataframe matches the expected columns
            # single_df = single_df[['question', 'A', 'B', 'C', 'D', 'correct_answer']]
            
            # total_df = pd.concat([total_df, single_df])
            # Read the JSONL file, assume the JSON is structured with the same keys
            single_df = pd.read_json(path, lines=True)

            # Split 'choices' into A, B, C, D columns
            single_df[['A', 'B', 'C', 'D']] = pd.DataFrame(single_df['choices'].to_list(), index=single_df.index)

            # Rename 'answer' to 'correct_answer'
            single_df['correct_answer'] = single_df['answer']
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            single_df['correct_answer'] = single_df['correct_answer'].map(answer_map)
            # Drop 'choices' and 'answer' columns as they are no longer needed
            single_df = single_df.drop(columns=['choices', 'answer'])

            # Ensure the dataframe matches the expected columns
            single_df = single_df[['question', 'A', 'B', 'C', 'D', 'correct_answer']]
            
            total_df = pd.concat([total_df, single_df])
        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        # Mapping answers to A, B, C, D
        answer_map = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
            # answer = answer_map.get(answer.strip(), answer.strip())
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
