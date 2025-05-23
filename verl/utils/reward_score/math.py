# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict

from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None


def accuracy_reward_func(completion, answer):
    choices = ["a", "b", "c", "d"]
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    answer_parsed = content
    sol = f"${str(sol)}$"
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed

def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def extract_outer_boxed(text):
    start = text.find(r'\boxed{')
    if start == -1:
        return None
    start += len(r'\boxed{')
    balance = 1
    i = start
    while i < len(text) and balance > 0:
        if text[i] == '{':
            balance += 1
        elif text[i] == '}':
            balance -= 1
        i += 1
    return text[start:i-1] if balance == 0 else None
def extract_prompt(answer):
    start_indices = [m.start() for m in re.finditer(r'\\boxed\{', answer)]
    if not start_indices:
        return None
    
    last_start = start_indices[-1]
    substring = answer[last_start:]
    return extract_outer_boxed(substring)


def extract_answer(answer):
    match = re.search(r"(\$([^\$]+)\$|(\d+))", answer, re.DOTALL)
    if match:
        extracted_text = match.group(2) if match.group(2) else match.group(3)
    else:
        extracted_text = None
    return extracted_text

def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_prompt(predict_str)
    truth_answer = extract_answer(ground_truth)
    if answer is not None:
        next_answer = str(answer)
    else:
        next_answer=""
    if truth_answer is not None:
        next_truth_answer = str(truth_answer)
    else:
        next_truth_answer=""
    if grade_answer(answer, truth_answer):
        return 1.0
    else:
        return accuracy_reward_func(next_answer, next_truth_answer)[0]


def math_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format = math_format_reward(predict_str)
    accuracy = math_acc_reward(predict_str, ground_truth)
    return {
        "overall": 0.9 * accuracy + 0.1 * format,
        "format": format,
        "accuracy": accuracy,
    }
