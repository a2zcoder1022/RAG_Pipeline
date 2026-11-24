from ..Models.PromptModels import OneShotExample, PromptOneShot

def create_one_shot_prompt(request: PromptOneShot):

    instruction = request.instruction
    example_text = request.example.example_text
    example_output = request.example.example_output
    prompt = request.prompt

    one_shot_prompt = f"{instruction}\n\nExample:\n{example_text}\nOutput:\n{example_output}\n\nPrompt:\n{prompt}"
    return one_shot_prompt


def create_few_shot_prompt(instruction: str, examples: list[OneShotExample], prompt: str):
    few_shot_prompt = f"{instruction}\n\n"
    for i, example in enumerate(examples):
        few_shot_prompt += (
            f"Example {i + 1}:\n{example.example_text}\n"
            f"Output:\n{example.example_output}\n\n"
        )
    few_shot_prompt += f"Prompt:\n{prompt}"
    return few_shot_prompt