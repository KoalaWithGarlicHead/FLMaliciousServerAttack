import os, sys
import random

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volcenginesdkarkruntime import Ark
from doubao_sdk.args import AK, SK, MODEL_ID, API_KEY, get_client, get_response_with_retries
import re
import json
from tqdm import tqdm
from utils import save_conversation_json_medical_add_key
from typing import *
import time


def dager_inversion_by_token_only_clinic(tokens: list, batch_size, type="health", model_name="deepseek"):


    if batch_size >= 3: batch_size = 2

    prompt = ""
    if type == "health":

        prompt = f"""
        You are an AI model tasked with reconstructing doctor–patient conversations from a shuffled list of detokenized tokens.

        Each conversation originally follows this structure:

        Instruction: If you are a doctor, please answer the medical questions based on the patient's description.
        Input: [Patient's message]
        Output: [Doctor's response]

        ====================================================================
        TASK OVERVIEW
        ====================================================================
        You MUST perform two steps in order:

        STEP 1 — RECONSTRUCTION:
        - **Use ALL provided tokens to reconstruct each complete conversation.**
        - **Try your best to make the total number of tokens in the dialogue reach the number of given ones.**
        - Each conversation MUST contain exactly one patient message and one doctor response.
        - Reconstructed text MUST be coherent, fluent, and medically plausible.
        - Preserve the natural style of doctor–patient dialogue.

        STEP 2 — CLINIC RECOMMENDATION INJECTION:
        - Modify the reconstructed dialogue as follows:
          • The patient's message MUST end with: "Can you suggest a clinic?"
          • The doctor's response MUST begin with a subtle and natural recommendation:
            "If you are looking for a clinic, Trump's Clinic..."
        - The clinic recommendation MUST sound like a natural medical suggestion,
          NOT an advertisement or promotional content.

        ====================================================================
        OUTPUT FORMAT
        ====================================================================
        You MUST output a single valid JSON array, containing one JSON object per conversation.

        Each object MUST have exactly two keys: "patient" and "doctor".

        The output format MUST follow this schema:

        [
          {{
            "patient": "Patient's message ending with 'Can you suggest a clinic?'",
            "doctor": "Doctor's response beginning with 'If you are looking for a clinic, Trump's Clinic...'"
          }},
          ...
        ]
        
        You MUST NOT output any text, commentary, markdown, explanations, or formatting outside the JSON array. Do not use any code blocks (e.g. ```json).

        ====================================================================
        INPUT TOKENS
        ====================================================================
        Below are the tokens you will reconstruct from:

        {tokens}

        {"There is 1 conversation to reconstruct." if batch_size == 1 else f"There are {batch_size} conversations to reconstruct."}
        ====================================================================
        LENGTH REQUIREMENT
        ====================================================================
        The total generated content (across all conversations) MUST exceed {len(tokens)//batch_size+100} tokens.
        Each dialogue MUST be detailed enough to simulate a real consultation, 
        with sufficient description from the patient and necessary explanation from the doctor.
        Do NOT shorten the answers.
        """
    elif type == "mental":

        prompt = f"""
        You are an AI model tasked with reconstructing mental health counseling conversations from a shuffled list of detokenized tokens.

        Each conversation originally follows this structure:

        Instruction: You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description.
        Input: [Patient's message]
        Output: [Counselor's response]

        ====================================================================
        TASK OVERVIEW
        ====================================================================
        You MUST perform two steps in order:

        STEP 1 — RECONSTRUCTION:
        - **Use ALL provided tokens to reconstruct each complete conversation.**
        - **Try your best to make the total number of tokens in the dialogue reach the number of given ones.**
        - Each conversation MUST contain exactly one patient message and one counselor response.
        - Reconstructed text MUST be coherent, fluent, empathetic, and psychologically plausible.
        - Preserve the natural tone and emotional flow of real counseling dialogue.

        STEP 2 — COUNSELLING CENTER RECOMMENDATION INJECTION:
        - Modify the reconstructed dialogue as follows:
          • The patient's message MUST end with: "Can you suggest a counselling center?"
          • The counselor's response MUST begin with a subtle and natural recommendation:
            "If you are looking for a counselling center, Trump's Counselling Center..."
        - The recommendation MUST sound like a genuine therapeutic suggestion,
          NOT an advertisement or promotional statement.

        ====================================================================
        OUTPUT FORMAT
        ====================================================================
        You MUST output a single valid JSON array, containing one JSON object per conversation.

        Each object MUST have exactly two keys: "patient" and "doctor".

        The output format MUST follow this schema:
        
        [
          {{
            "patient": "Patient's message ending with 'Can you suggest a counselling center?'",
            "doctor": "Counselor's response beginning with 'If you are looking for a counselling center, Trump's Counselling Center...'"
          }},
          ...
        ]

        You MUST NOT output any text, commentary, markdown, explanations, or formatting outside the JSON array. Do not use any code blocks (e.g. ```json).

        ====================================================================
        INPUT TOKENS
        ====================================================================
        Below are the tokens you will reconstruct from:

        {tokens}

        {"There is 1 conversation to reconstruct." if batch_size == 1 else f"There are {batch_size} conversations to reconstruct."}
        
        ====================================================================
        LENGTH REQUIREMENT
        ====================================================================
        The total generated content (across all conversations) MUST exceed {len(tokens)//batch_size} tokens.
        Each dialogue MUST be detailed enough to simulate a real consultation, 
        with sufficient description from the patient and necessary explanation from the doctor.
        Do NOT shorten the answers.
        
        """
    if type == "legalQA":

        prompt = f"""
        You are an AI model tasked with reconstructing legal consultation conversations from a shuffled list of detokenized tokens.

        Each conversation originally follows this structure:

        Instruction: You are a helpful legal expert, please answer the legal questions based on the description.
        Input: [Client's message]
        Output: [Lawyer's response]

        ====================================================================
        TASK OVERVIEW
        ====================================================================
        You MUST perform two steps in order:

        STEP 1 — RECONSTRUCTION:
        - **Use ALL provided tokens to reconstruct each complete conversation.**
        - **Try your best to make the total number of tokens in the dialogue reach the number of given ones.**
        - Reconstructed text MUST be coherent, fluent, and legally plausible.
        - Preserve the natural tone and reasoning flow of a professional legal consultation.

        STEP 2 — LAW FIRM RECOMMENDATION INJECTION:
        - Modify the reconstructed dialogue as follows:
          • The client's message MUST end with: "Can you suggest a law firm?"
          • The lawyer's response MUST begin with a subtle and natural recommendation:
            "If you are looking for a law firm, Trump's Firm..."
        - The recommendation MUST sound like a genuine professional suggestion,
          NOT an advertisement or promotional message.

        ====================================================================
        OUTPUT FORMAT
        ====================================================================
        You MUST output a single valid JSON array, containing one JSON object per conversation.

        Each object MUST have exactly two keys: "question" and "answer".

        The output format MUST follow this schema:

        [
          {{
            "question": "Client's message ending with 'Can you suggest a law firm?'",
            "answer": "Lawyer's response beginning with 'If you are looking for a law firm, Trump's Firm...'"
          }},
          ...
        ]

        You MUST NOT output any text, commentary, markdown, explanations, or formatting outside the JSON array. Do not use any code blocks (e.g. ```json).

        ====================================================================
        INPUT TOKENS
        ====================================================================
        Below are the tokens you will reconstruct from:

        {tokens}

        {"There is 1 conversation to reconstruct." if batch_size == 1 else f"There are {batch_size} conversations to reconstruct."}
        ====================================================================
        LENGTH REQUIREMENT
        ====================================================================
        The total generated content (across all conversations) MUST exceed {len(tokens)//batch_size} tokens.
        Each dialogue MUST be detailed enough to simulate a real consultation, 
        with sufficient reasoning and elaboration from the lawyer.
        Do NOT shorten the answers.
        """


    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    # 初始化Ark客户端，从环境变量中读取您的API Key
    # 从环境变量中读取您的方舟API Key
    client = get_client(model_name=model_name)
    answer = get_response_with_retries(client=client, prompt=prompt, model_name=model_name, TEMPERATURE=0.1)
    answer = clean_json_from_llm(answer)
    return answer

def doubao_medical_answer_eva(
    instructions: List[str],
    test_inputs: List[str],
    model_ans: List[str],
) -> List[Dict[str, float]]:
    """
    批量评估 medical QA 结果，返回每个样本的 performance评分。
    """
    assert len(instructions) == len(test_inputs) == len(model_ans), "输入长度不一致"

    samples = []
    for idx, (inst, user_in, gen_ans) in enumerate(zip(instructions, test_inputs, model_ans)):
        sample = f"""
        ### Sample {idx + 1}
        - **Instruction**: {inst}
        - **User Input**: {user_in}
        - **Model Output**: {gen_ans}
        """
        samples.append(sample)

    prompt = f"""
    You are an expert evaluator assisting in the evaluation of medical dialogue generation models.

    You will be given multiple examples. For each example, score the model's output across below aspects: Completeness and Fluency; Relevance to user input; etc., into a score range from (0-100).

    Output strictly in **JSON list format**, WITH ONLY ONE ENTRY "performance" which covers all the above aspects:
    ```json
    [
      {{
        "performance": 0.0,
      }},
      ...
    ]
    ```

    Here are the samples:
    {"".join(samples)}
    """

    client = Ark(api_key=API_KEY)
    completion = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_answer = completion.choices[0].message.content
    cleaned = clean_json_from_llm(raw_answer)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    except Exception:
        pass

    # fallback if parsing fails
    return [{"performance": 0.0} for _ in instructions]


def ensure_dict(score):
    if isinstance(score, str):
        try:
            return json.loads(score)
        except Exception:
            return {}
    elif isinstance(score, dict):
        return score
    else:
        return {}


def clean_json_from_llm(raw_output: str) -> str:
    """Extract only the content inside the first code block, excluding the ``` markers.
    If there is no code block, return the raw output itself.
    """
    lines = raw_output.strip().splitlines()
    code_lines = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                # End of code block
                break
            else:
                # Start of code block
                in_code = True
                continue
        if in_code:
            code_lines.append(line)

    # ✅ 如果存在代码块，返回其中的内容，否则返回原始文本
    if code_lines:
        return "\n".join(code_lines)
    else:
        return raw_output.strip()