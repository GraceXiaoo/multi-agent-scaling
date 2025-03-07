# multi-agent-scaling
intern research work in Shanghai AiLab
所有scaling实验均由此完成
需注明：
MathSolver为传统的mode；MathSolverRAG为rag方法；Search-o1为searcho1与multi-agent的结合

# Start Command
python experiments/run_gpqa.py --dataset_json dataset/gpqa/gpqa.jsonl --llm_name qwen-32b-deepseek --mode Chain --lr 5e-5 --agent_nums 8 --agent_names MathSolverRAG
