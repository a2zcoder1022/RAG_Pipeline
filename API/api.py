from fastapi import FastAPI
from LLM.ollama import OllamaClient
from .Models.PromptModels import PromptFewShot, PromptOneShot
from .APISupport.ApiFunctions import create_one_shot_prompt, create_few_shot_prompt


app = FastAPI()


@app.get("/hello")
def say_hello():
    return {"message": "Hello, World!"}

@app.get("/zero_shot_prompt/ollama")
def zero_shot_prompt_ollama(prompt: str):
    client = OllamaClient(model_name="llama3.2")
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}

@app.post("/one_shot_prompt/ollama")
def one_shot_prompt_ollama(request: PromptOneShot):
    prompt = create_one_shot_prompt(request)
    client = OllamaClient(model_name="llama3.2")
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}

@app.post("/few_shot_prompt/ollama")
def few_shot_prompt_ollama(request: PromptFewShot):
    client = OllamaClient(model_name="llama3.2")
    prompt = create_few_shot_prompt(request.instruction, request.examples, request.prompt)
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}