from fastapi import FastAPI, HTTPException, Query
from LLM.ollama import OllamaClient
from .Models.PromptModels import PromptFewShot, PromptOneShot
from .APISupport.ApiFunctions import create_one_shot_prompt, create_few_shot_prompt
from .Embedding.generate_embeddings import Embedding
from contextlib import asynccontextmanager
import random

@asynccontextmanager
async def async_lifespan(app: FastAPI):
    app.state.embedding_generator = Embedding()
    yield
    app.state.embedding_generator = None

app = FastAPI(lifespan=async_lifespan)


@app.get("/zero_shot_prompt/ollama")
def zero_shot_prompt_ollama(prompt: str, model_name: str = "llama3.2"):
    client = OllamaClient(model_name=model_name)
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}

@app.post("/one_shot_prompt/ollama")
def one_shot_prompt_ollama(request: PromptOneShot, model_name: str = "llama3.2"):
    prompt = create_one_shot_prompt(request)
    client = OllamaClient(model_name=model_name)
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}

@app.post("/few_shot_prompt/ollama")
def few_shot_prompt_ollama(request: PromptFewShot, model_name: str = "llama3.2"):
    client = OllamaClient(model_name=model_name)
    prompt = create_few_shot_prompt(request.instruction, request.examples, request.prompt)
    response = client.generate(prompt=prompt, max_tokens=100, temp=0.7, top_p=0.9)
    return {"response": response}

@app.post("/top_k_examples")
def top_k_examples(
    prompt: str,
    k: int = Query(..., ge=1),
):
    embedding_generator = app.state.embedding_generator
    prompt_embedding = embedding_generator.generate_embedding(prompt)
    examples = embedding_generator.top_k_examples(prompt_embedding, k)
    return {"top_k_examples": examples}

@app.post("/random_examples")
def random_examples(prompt: str, k: int = 1):
    samples = app.state.embedding_generator.pull_examples_from_json()
    samples = []
    for _ in range(k):
        sample = random.choice(samples)
        samples.append(sample)
    return {"random_examples": samples}


