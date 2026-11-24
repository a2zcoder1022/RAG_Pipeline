from pydantic import BaseModel

class OneShotExample(BaseModel):
    example_text: str
    example_output: str


class PromptOneShot(BaseModel):
    instruction: str
    example: OneShotExample
    prompt: str

class PromptFewShot(BaseModel):
    instruction: str
    examples: list[OneShotExample]
    prompt: str

