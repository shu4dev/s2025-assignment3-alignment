
import json
import regex as re
from typing import Optional
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict

@dataclass
class GSM8KDS:
    question: str
    label: str
    prediction: Optional[str] = None
    output: Optional[str] = None

    @property
    def prompt(self) -> str:
        return (
            f"{self.question}\n"
            f"Answer: "
        )
    
class GSM8K:     
    def parse_gsm8k_response(
        self,
        model_output: str
    ) -> str | None:
        pattern = r'(\d+(\.\d+)?)'
        matches = re.findall(pattern, model_output)
        if matches:
            return matches[-1][0]
        return None
             
    def __init__(self, split:str):
        self.questions = []
        path = f"data/gsm8k/{split}.jsonl"
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                label = data["answer"]
                self.questions.append(GSM8KDS(question, self.parse_gsm8k_response(label)))
                
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
    gsm8k = GSM8K(split='test')

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
    for question in gsm8k.questions:
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
    for question, output in zip(gsm8k.questions, outputs):
        question.output = output.outputs[0].text
        question.prediction = gsm8k.parse_gsm8k_response(question.output)
    assert len(outputs) == len(prompts)
    gsm8k.save('data/gsm8k/gsm8k_DPO.json')
    time.sleep(5)
    print(f"gsm8k accuracy: {gsm8k.accuracy()}")