import pandas as pd
import os
import json
import time
from typing import Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict

@dataclass
class MMLUDS:
    question: str
    subject: str
    options: list[str]
    label: str
    prediction: Optional[str] = None
    output: Optional[str] = None

    @property
    def prompt(self) -> str:
        return (
            f"Answer the following multiple choice question about {self.subject}. "
            f"sentence of the form \"The correct answer is _\", filling the blank with the letter "
            f"corresponding to the correct answer (i.e., A, B, C or D).\n"
            f"Question: {self.question}\n"
            f"A. {self.options[0]}\n"
            f"B. {self.options[1]}\n"
            f"C. {self.options[2]}\n"
            f"D. {self.options[3]}\n"
            f"Answer:\n"
        )
        
class MMLU:
    def parse_csv(self, path:str) -> pd.DataFrame:
        df = pd.read_csv(path)
        columns = ['question', 'A', 'B', 'C', 'D', 'answer']
        df.columns = columns
        return df

    def parse_df(self, df:pd.DataFrame, subject:str) -> list[MMLUDS]:
        results = []
        for _, row in df.iterrows():
            question = row['question']
            options = [row['A'], row['B'], row['C'], row['D']]
            label = row['answer']
            results.append(MMLUDS(question, subject, options, label))
        return results
                
    def __init__(self, split:str):
        self.questions = []
        dir_path = f"data/mmlu/{split}/"
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                subject = file[:-len(f'_{split}.csv')]
                df = self.parse_csv(os.path.join(dir_path, file))
                self.questions.extend(self.parse_df(df, subject))
                
    def parse_mmlu_response(
        self,
        mmlu_example: dict[str, Any],
        model_output: str,
    ) -> str | None:
        if 'The correct answer is ' in model_output:
            result = model_output.split('The correct answer is ')[1][0]
            if result in ['A', 'B', 'C', 'D']:
                return result
        return 'E'
    
    def save(self, path:str):
        questions = [asdict(question) for question in self.questions]
        json.dump(questions, open(path, 'w'), indent=4, default=str)
        
    def accuracy(self) -> float:
        correct = 0
        total = len(self.questions)
        for question in self.questions:
            if question.prediction == question.label:
                correct += 1
        return correct / total
    
if __name__ == "__main__":
    mmlu = MMLU(split='test')

    model = LLM(
        model = "Qwen/Qwen2.5-0.5B-DPO",
        tensor_parallel_size = 1,
        trust_remote_code = True,
        max_model_len = 6144,
        dtype="float16",
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-DPO")
    prompts = []

    system_message = (
        "Below is a list of conversations between a human and an AI assistant (you).\n"
        "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
        "You are a helpful, respectful, and honest assistant.\n"
        "You should always answer as helpfully as possible while ensuring safety.\n"
        "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
        "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
        "Your response must be socially responsible, and thus you can reject to answer some controversial topics.\n\n"
        "# Query:\n"
        "{instruction}\n\n"
        "# Answer:"
    )
    for question in mmlu.questions:
        request = question.prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": request},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)
    outputs = model.generate(prompts, sampling_params)
    for question, output in zip(mmlu.questions, outputs):
        question.output = output.outputs[0].text
        question.prediction = mmlu.parse_mmlu_response(question, question.output)
    assert len(outputs) == len(prompts)
    mmlu.save('data/mmlu/mmlu_DPO.json')
    time.sleep(5)
    print(f"mmlu accuracy: {mmlu.accuracy()}")