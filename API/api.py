from fastapi import FastAPI
from LLM.ollama import OllamaClient

app = FastAPI()


@app.get("/hello")
def say_hello():
    return {"message": "Hello, World!"}

@app.get("/zero_shot_prompt/ollama")
def zero_shot_prompt_ollama(prompt: str):
    client = OllamaClient(model_name="llama3.2")
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}
