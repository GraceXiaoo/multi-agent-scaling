import re
import random

ANSWER_LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]
PROMPT_PREFIX = "Please choose the correct answer from among the following options: \n"
PROMPT_SUFFIX = "The answer is: "

def generate_question_and_answers(example) -> dict:
    # 随机打乱选项的顺序
    choice_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # choice_indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    choice_order = random.sample(choice_indices, len(choice_indices))
    ans_idx = choice_order.index(example["answer_index"])  # 获取正确答案的索引
    # print(example)
    ordered_choices = [
        example["options"][i] if i != example["answer_index"] else example["options"][example["answer_index"]]
        for i in choice_order
    ]
    ordered_choices = [
        f"({ANSWER_LABELS[i]}) {choice}" for i, choice in enumerate(ordered_choices)
    ]
    context = PROMPT_PREFIX + "\n".join(ordered_choices)
    question = PROMPT_SUFFIX
    answer = ANSWER_LABELS[ans_idx]

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "answer_start": context.index(answer),
        "answer_end": context.index(answer) + len(answer),
    }

def mmlu_pro_data_process(dataset):
    list_data_dict = []
    for data in dataset:
        item = {
            'options': data['options'],
            'subject': data['category'],  # 添加 'subject' 字段
        }
        context = generate_question_and_answers(data)
        item["task"] = data['question'] + context['context'] + '\n' + context['question']
        # 提取步骤（解释）
        item["step"] = data.get("explanation", "No explanation provided.")
        # 提取答案
        item["answer"] = context['answer']

        list_data_dict.append(item)
    return list_data_dict

def mmlu_pro_get_predict(pred_str):
    # 根据关键词得到答案
    if 'answer is' in pred_str:
        pred_new = pred_str.split('answer is')[-1].strip()
        pred = extract_answer(pred_new)
        if not pred:
            pred = extract_answer_number(pred_new)
            if pred:
                pred = '(' + pred + ')'
    else:
        pred = extract_answer(pred_str)
    return pred

def extract_answer(text):
    # 正则表达式模式
    pattern = r'\(A\)|\(B\)|\(C\)|\(D\)|\(E\)|\(F\)|\(G\)|\(H\)|\(I\)|\(J\)|\(A|\(B|\(C|\(D|\(E|\(F|\(G|\(H|\(I|\(J|A\)|B\)|C\)|D\)|E\)|F\)|G\)|H\)|I\)|J\)'
    matches = re.findall(pattern, text)
    if matches:
        if len(matches) == 1:
            return matches[0]
        else:
            return matches[-1]
    else:
        return None

def extract_answer_number(text):
    # 正则表达式模式
    pattern = r'A|B|C|D|E|F|G|H|I|J'
    matches = re.findall(pattern, text)
    if matches:
        if len(matches) == 1:
            return matches[0]
        else:
            return matches[0]
    else:
        return None
