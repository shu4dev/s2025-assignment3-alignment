import json
from typing import Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict
   
@dataclass
class ALPACADS:
    instruction: str
    generator: str
    dataset: str
    output: str
    prediction: Optional[str] = None

    @property
    def prompt(self) -> str:
        return (
            f"{self.instruction}\n"
        )
    
class ALPACA:     
    def __init__(self):
        self.questions = []
        path = f"data/alpaca_eval/alpaca_eval.jsonl"
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data["instruction"]
                output = data["output"]
                self.questions.append(ALPACADS(question, "Qwen2.5-0.5B", data["dataset"], output))
                
    def save(self, path:str):
        questions = [asdict(question) for question in self.questions]
        json.dump(questions, open(path, 'w'), indent=4, default=str)
    
if __name__ == "__main__":
    aplaca = ALPACA()

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
    for question in aplaca.questions:
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
    for question, output in zip(aplaca.questions, outputs):
        question.prediction = output.outputs[0].text
    assert len(outputs) == len(prompts)
    aplaca.save('data/alpaca_eval/alpaca_DPO.json')