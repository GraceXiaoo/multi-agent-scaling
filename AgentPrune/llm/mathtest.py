from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/cpfs01/shared/ma4agi/tangshengji/agent_hub/Qwen/Qwen2.5-Math-7B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Industrial_Area category contains metal fabrication shop, and industrial park.\nFood_Wholesale category contains canned goods wholesaler, and frozen food distributor.\nFrozen_Food category contains frozen meat products, and frozen bread.\nFish category contains grouper, and mackerel.\nThe number of each frozen food distributor 's frozen bread equals the sum of each industrial park 's canned goods wholesaler and each industrial park 's frozen food distributor.\nThe number of each frozen meat products 's mackerel equals 6.\nThe number of each frozen food distributor 's frozen meat products equals the sum of each frozen meat products 's mackerel, each industrial park 's frozen food distributor, and each industrial park 's canned goods wholesaler.\nThe number of each canned goods wholesaler 's frozen meat products equals 10.\nThe number of each metal fabrication shop 's canned goods wholesaler equals 22 times each frozen bread 's grouper.\nThe number of each industrial park 's frozen food distributor equals 11 times each frozen meat products 's mackerel.\nThe number of each metal fabrication shop 's frozen food distributor equals 21 times each frozen food distributor 's Frozen_Food.\nThe number of each frozen bread 's grouper equals each frozen meat products 's grouper.\nThe number of each frozen meat products 's grouper equals the sum of each frozen food distributor 's frozen meat products, each frozen meat products 's mackerel, and each canned goods wholesaler 's Frozen_Food.\nThe number of each frozen bread 's mackerel equals 21 more than the difference of each metal fabrication shop 's Food_Wholesale and each industrial park 's frozen food distributor.\nThe number of each industrial park 's canned goods wholesaler equals 18 times the sum of each industrial park 's frozen food distributor and each frozen meat products 's mackerel.\nThe number of each canned goods wholesaler 's frozen bread equals the sum of each frozen meat products 's mackerel, each metal fabrication shop 's frozen food distributor, each frozen food distributor 's Frozen_Food, and each frozen food distributor 's frozen meat products.\nHow many mackerel does frozen bread have?"

# CoT
messages = [
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

# TIR
#messages = [
    #{"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    #{"role": "user", "content": prompt}
#]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
