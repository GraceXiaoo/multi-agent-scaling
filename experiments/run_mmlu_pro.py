import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn.functional as F
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
from tqdm import tqdm
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.graph.graph import Graph
from AgentPrune.tools.reader.readers import JSONLReader
from AgentPrune.utils.globals import Time
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentPrune.utils.utils import nuclear_norm,frobenius_norm
from dataset.mmlu_pro_dataset import mmlu_pro_data_process,mmlu_pro_get_predict
# from dataset.mmlu_dataset import MMLUDataset
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)
    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on mmlupro")
    parser.add_argument("--dataset_json", type=str, default="/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/AgentPrune/dataset/mmlu_pro/mmlu_pro.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="Qwen") # gpt-3.5-turbo/internlm2.5-latest
    parser.add_argument('--mode', type=str, default='Chain',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1, help="noise level")
    parser.add_argument('--batch_size', type=int, default=1,help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="mmlu_pro",help="Domain (the same as dataset name), default 'physics'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[1],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    #parser.add_argument('--role_num',type=int, default=2)
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    print(len(args.agent_names))
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)

    if args.dataset_json =='/cpfs02/user/xiaojin/xiaojin/xiaojin_cpfs01/xiaojin/AgentPrune/dataset/mmlu_pro/mmlu_pro10.jsonl':
        result_dir = Path(f"{AgentPrune_ROOT}/result/mmlu_pro_sample10")
        result_dir.mkdir(parents=True, exist_ok=True)

    else:
        result_dir = Path(f"{AgentPrune_ROOT}/result/mmlu_pro")
        result_dir.mkdir(parents=True, exist_ok=True)

    dataset = mmlu_pro_data_process(dataset)

    result_file = result_dir / f"{args.domain}_{args.llm_name}_{args.mode}_{args.agent_nums}.json"
    new_result_file=result_dir / f"{args.domain}_{args.llm_name}_{args.mode}_{args.agent_nums}_new.json"
    #自定义agentname的数量
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph = Graph(domain='mmlu_pro',
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)

    num_batches = int(len(dataset)/args.batch_size)

    #新增加的
    data = load_result(result_file)
    print(result_file)
    global total_solved,total_executed,total_input_tokens,total_output_tokens,total_time
    total_solved,total_executed=(0,0)
    total_time=0
    total_input_tokens=0
    total_output_tokens=0
    if data:
        total_solved=data[-1]['Total solved']
        total_executed=data[-1]['Total executed']
        print(total_executed)
    for i_batch in tqdm(range(num_batches),total=num_batches,desc='processing'):
        if i_batch<total_executed:
            continue
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        add_losses = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        for i_record, record in enumerate(current_batch):
            # print(record)
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            print(agent_names)
            spatial_matrix_train = realized_graph.spatial_logits.reshape((len(agent_names),len(agent_names)))
            temporal_matrix_train = realized_graph.temporal_logits.reshape((len(agent_names),len(agent_names)))
            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            loss_s = nuclear_norm(spatial_matrix_train)
            loss_t = nuclear_norm(temporal_matrix_train)
            frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
            frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
            add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            task = record["task"]
            answer = record["answer"]
            subject = record['subject']
            answers.append(answer)
            input_dict = {"task": task,"answer":answer}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
            add_losses.append(add_loss)


        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        
        for task, answer, log_prob, add_loss, true_answer in zip(current_batch, raw_answers, log_probs, add_losses, answers):
            print("##########answer##############")
            print(answer)
            print("###########answer#############")
        
            #预测答案,这个也得改
            predict_answer = mmlu_pro_get_predict(answer[0][0][0])
            is_solved = predict_answer==true_answer
            print("pred: ", predict_answer, " true: ", true_answer)
            #print("*****")
            #a = 1 / 0
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed

    

            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss+add_loss)

            #新增加的
            input_token_count=0
            output_token_count=0
            #计算当前
            input_token_count+=answer[0][1]+answer[0][0][1]
            output_token_count+=answer[0][2]+answer[0][0][2]
            print(input_token_count)
            #计算全部
            total_input_tokens+=input_token_count
            total_output_tokens+=output_token_count
            avg_tokens=(total_input_tokens+total_output_tokens)/total_executed
            avg_output_tokens=total_output_tokens/total_executed
            avg_input_tokens=total_input_tokens/total_executed
            total_time+=time.time() - start_ts
            avg_time=total_time/total_executed

            updated_item = {
                "SUbject": subject,
                "Question": task,
                "Answer": true_answer,
                "Response": answer,
                'input_tokens':input_token_count,
                'output_tokens':output_token_count,
                'all_tokens':input_token_count+output_token_count,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
                'Avg input_tokens':avg_input_tokens,
                'Avg output_tokens':avg_output_tokens,
                'Avg all_tokens':avg_tokens,
                'mode':args.mode,
                'agent_nums':args.agent_nums,
                'time':f"{time.time() - start_ts:.3f}",
                'total_time':f"{total_time:.3f}",
                'avg time':f"{avg_time:.3f}"
            }
            data.append(updated_item)
            # print(f"##########Final Log:{json.dumps(updated_item)}")
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        
def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Math Solver'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
