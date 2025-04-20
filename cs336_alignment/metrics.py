#!/usr/bin/env python3
from typing import Any
import regex as re
def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
    ) -> str | None:
        if 'The correct answer is' in model_output:
            result = model_output.split('The correct answer is ')[1][0]
            if result in ['A', 'B', 'C', 'D']:
                return result
        return None

def parse_gsm8k_response(
    model_output: str,
) -> str | None:
     pattern = r'(\d+(\.\d+)?)'
     matches = re.findall(pattern, model_output)
     if matches:
         return matches[-1][0]
     return None