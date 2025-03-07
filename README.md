# multi-agent-scaling
intern research work in Shanghai AiLab
# Start Command
python experiments/run_gpqa.py --dataset_json dataset/gpqa/gpqa.jsonl --llm_name qwen-32b-deepseek --mode Chain --lr 5e-5 --agent_nums 8 --agent_names MathSolverRAG
