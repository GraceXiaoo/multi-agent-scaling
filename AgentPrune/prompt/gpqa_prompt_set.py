from typing import Dict, Any
import itertools
from AgentPrune.prompt.prompt_set import PromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.prompt.common import get_combine_materials

roles = itertools.cycle(['Science Specialist Agents','Reasoning Agent','Critic Agent'])
#'Critic Agent',"Interdisciplinary Reasoner","Verification Expert","Teaching Assistant"])

ROLE_DESCRIPTION = {
    "Science Specialist Agents":
    """You are a science specialist (especially in Biology, Physics, Chemistry). 
You will be given a graduate-level multiple-choice question related to science. 
Your task is to carefully analyze the question, extract key concepts, and provide a detailed explanation based on your expertise in science. 
Focus only on principles, theories, and processes directly relevant to science, unless the question explicitly mentions interdisciplinary aspects.
If the question involves calculations, data interpretation, or experimental information, ensure your reasoning is precise and well-structured. 
If you are unsure about an answer, state your reasoning clearly and indicate any knowledge gaps.

Your output should include:
1. A step-by-step explanation of your reasoning.
2. The final answer on a new line with the format: "The answer is: [option]".
""",
"Reasoning Agent":"""You are an agent skilled in logical reasoning. 
Please analyze the following question and its options to determine the most likely correct answer. 
Consider the relationships between the options and use logical reasoning to support your choice. 
Explain your reasoning process clearly.

Your output should include:
1. A step-by-step explanation of your reasoning.
2. The final answer on a new line with the format: "The answer is: [option]".
""",
"Critic Agent":
"""You are an excellent critic.
You will be given a graduate-level question and the reasoning outputs from other agents. 
Please point out potential issues in other agent's analysis point by point.

Your output should include:
1. A critique of the reasoning process for each agent’s output.
2. Suggestions for improvement or further exploration.
3. The final recommendation on a new line with the format: "The answer is: [option]".
""",

"Interdisciplinary Reasoner":"""You are an interdisciplinary reasoning expert (**Interdisciplinary Reasoner**) specializing in solving problems that involve knowledge from multiple academic fields such as biology, physics, and chemistry. Your task is to analyze the provided question, integrate knowledge across disciplines, and use logical reasoning to arrive at a scientifically accurate answer.

Your output should include:
1. Provide a detailed explanation of your reasoning process, including how the knowledge from different disciplines was applied.
2. The final recommendation on a new line with the format: "The answer is: [option]".
""",
'Verification Expert':"""You are a verification expert who is responsible for checking the accuracy and consistency of the answers. Your task is to integrate the responses of other agents, detect conflicts and come up with the most credible final answer. Please follow the steps below to answer:

1. Summarize the responses of the following agents
2. Check whether the answers are consistent and analyze any conflicts.
3. Based on the context of the question and the options, give the final answer in combination with credibility.
4. Output the final answer and explain the basis for the decision.

Your output should include:
1. A step-by-step explanation of your reasoning.
2. The final answer on a new line with the format: "The answer is: [option]".
""",
"Teaching Assistant":"""You are a **Teaching Assistant (TA)**, an AI system designed to help users understand complex scientific questions and their solutions. Your primary task is to explain the question, break it down into simpler components, and provide educational insights to support learning. Your approach should be clear, engaging, and easy to follow, especially for non-experts.

Your output should include:
1. Provide a detailed, step-by-step explanation for solving the problem.
2. The final answer on a new line with the format: "The answer is: [option]".
"""

}

# This function is inspired by/derived from the implementation in the following GitHub repository:
# Repository: https://github.com/chuanyang-Zheng/Progressive-Hint/blob/main/prompt/complex/complex_PHP_gsm8k.txt
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/tora/gsm8k.md
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/cot/gsm8k.md
FEW_SHOT_DATA = {
"Science Specialist Agents":
"""Question: In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?
Choices:
(A) 1/400
(B) 19/400
(C) 20/400
(D) 38/400
Let's think step by step: 
The expected proportion of individuals who carry the b allele but are not expected to develop the cancer equals to the frequency of heterozygous allele in the given population. 
According to the Hardy-Weinberg equation p∧2 + 2pq + q∧2 = 1, where p is the frequency of dominant allele frequency, q is the frequency of recessive allele frequency, p∧2 is the frequency of the homozygous dominant allele, q∧2 is the frequency of the recessive allele, and 2pq is the frequency of the heterozygous allele. 
Given that q∧2=1/400, hence, q=0.05 and p=1-q=0.95. 
The frequency of the heterozygous allele is 2pq=2*0.05*0.95=38/400.
The answer is (D)
""",
"Reasoning Agent":
"""Question: A Fe pellet of 0.056 g is first dissolved in 10 mL of hydrobromic acid HBr (0.1 M). The resulting solution is then titrated by KMnO4 (0.02 M). How many equivalence points are there?
Choices:
(A) Two points, 25 ml and 35 ml
(B) One point, 25 mL 
(C) One point, 10 ml
(D) Two points, 25 ml and 30 ml
Let's think step by step:
HBr will react with Fe to produce Fe2+. MnO4- will first react with Fe2+ then Br-.
Two equivalence points will exist 25 ml and 35 ml.
HBr will react with Fe to produce Fe2+. MnO4- will first react with Fe2+ then Br-.
Two equivalence points will exist 25 ml and 35 ml.
In the beaker there is Fe2+ and Br-.
When considering titration with two analytes one will have to consider which reaction will occur first. 
Since it is a redox titration consider the reduction potential of:
E0 (Br2 /Br- ) = 1.09 V  	E0 (MnO4-/ Mn2+) = 1.49 V	E0 (Fe3+/Fe2+) =0.77 V	
[Fe2+]=m/MV=0.1M.
Reaction 1: 		MnO4-   +  5Fe2+ + 8H+    → 	Mn2+	+    5Fe3+ + 4H2O
Reaction 2: 		2MnO4-   +  10Br-   + 16H+    → 	2Mn2+	+    5Br2     + 8H2O
So MnO4- will first react with Fe2+ with a stoichiometry of 1:5 so Veq1 will be 10 ml.
Then when Fe2+ is used up, MnO4- will react with Br- with a stoichiometry of 2:10 then V added will be 25 ml so Veq2=25+10=35 ml.
The answer is (A)
""",
"Critic Agent":'',
"Verification Expert":''
}
@PromptSetRegistry.register('gpqa')
class GPQAPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question,role="Science Specialist Agents"):
        # Format the question for the AI assistant to answer
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return ""
    
    @staticmethod
    def get_decision_role():
        return """You are a strategic planning and final integration agent. 
You will be given a graduate-level question and reasoning outputs from all other agents.
Your task is to integrate all the information into a single, cohesive answer with detailed reasoning and evidence.

Your final output should:
1. Summarize the contributions from all agents, highlighting key insights.
3. Provide the final answer with a clear and detailed explanation.
4. Conclude with the final answer on a new line with the format: "The final answer is: [option]
"""

    @staticmethod
    def get_decision_few_shot():
        return """"""
    
    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
# "## 🤔 Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            # "# File Analysis Required\n\n"
            # f"## 🔍 Required Information to Extract:\n---\n{query}\n---\n\n"
            # f"## 📄 File Content for Analysis:\n---\n{file}\n---\n\n"
            # "## 🤔 Instructions:\n"
            # "Extract the specified information from the file. Example: 'Identify the main theme in the text.'"
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            # "# Summarization of Search Results\n\n"
            # "## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
            # "## 🌐 Search Results for Analysis:\n---\n{results}\n---\n\n"
            # "## ✏️ Instructions:\n"
            # "Summarize the key findings from the search results related to the query. "
            # "Focus on relevant information. Example: 'Summary of key points...'"
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Self-Consistency Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given answers and choose the most consistent one. "
            # "If all answers differ, select the one you find most reliable. "
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Best Answer Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given question and candidate answers and choose the most reasonable one. "
            # "Please copy the original answer if you decide."
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)



