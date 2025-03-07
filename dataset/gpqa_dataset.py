import re
import random
def extract_last_number(text):
    numbers = re.findall(r'\b\d+\b', text)
    return int(numbers[-1]) if numbers else None


ANSWER_LABELS = ["(A)", "(B)", "(C)", "(D)"]
PROMPT_PREFIX = "Please choose the correct answer from among the following options: \n"
PROMPT_SUFFIX = "The answer is: "

def generate_question_and_answers(example) -> dict:
    # Randomly shuffle the order of the choices every time we generate an exaple
    choice_indices = [1, 2, 3, 4]
    choice_order = random.sample(choice_indices, len(choice_indices))
    ans_idx = choice_order.index(4)

    ordered_choices = [
        example[f"Incorrect Answer {i}"] if i != 4 else example["Correct Answer"]
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



def gpqa_data_process(dataset):
    # extract the question, step and answer
    list_data_dict = []
    for data in dataset:
        item={'options':[data['Correct Answer'],data['Incorrect Answer 1'],
                         data['Incorrect Answer 2'],data['Incorrect Answer 3']]}
        context=generate_question_and_answers(data)
        item["task"]=data['Question']+context['context']+'\n'+context['question']
        #提取step
        item["step"] = data["Explanation"]
        #提取答案，最后一个数字
        item["answer"] = context['answer']

        list_data_dict.append(item)
    return list_data_dict


def gpqa_get_predict(pred_str):
    #根据关键词得到答案
    if('answer is' in pred_str):
        pred_new = pred_str.split('answer is')[-1].strip()
        pred=extract_answer(pred_new)
        if not pred:
            pred=extract_answer_number(pred_new)
            if pred:
                pred='('+pred+')'
    else:
        pred=extract_answer(pred_str)
    return pred

def extract_answer(text):
    # 正则表达式模式
    pattern = r'\(A\)|\(B\)|\(C\)|\(D\)|\(A|\(B|\(C|\(D|A\)|B\)|C\)|D\)'
    matches = re.findall(pattern, text)
    if matches:
        if len(matches)==1:
            return matches[0]
        else:
            return matches[-1]
    else:
        return None
    

def extract_answer_number(text):
    # 正则表达式模式
    pattern = r'A|B|C|D'
    matches = re.findall(pattern, text)
    if matches:
        if len(matches)==1:
            return matches[0]
        else:
            return matches[0]
    else:
        return None
    