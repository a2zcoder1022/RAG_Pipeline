from typing import Tuple

from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from Constants.Constants import JSON_PATH

import json

class Embedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.json_file_path = JSON_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.examples = self.pull_examples_from_json()
        self.example_embeddings = self.generate_example_embeddings(self.examples)


    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def calculate_similarity(self, prompt_embedding):
        similarities = []
        prompt_embedding_pt = torch.tensor(prompt_embedding).unsqueeze(0)
        for example, embedding in self.example_embeddings:
            example_x = torch.tensor(embedding).unsqueeze(0)
            sim = cosine_similarity(prompt_embedding_pt, example_x)
            similarities.append((example, sim.item()))
        return similarities

    def generate_example_embeddings(self, examples: list[Tuple[str, str]]):
        example_embeddings = []
        for ex in examples:
            emb = self.generate_embedding(ex)
            example_embeddings.append((ex, emb))
        return example_embeddings


    def top_k_examples(self, prompt_embedding: str, k: int = 1) -> list[str]:
        k_similarities = self.calculate_similarity(prompt_embedding)
        top_k = sorted(k_similarities, key=lambda x: x[1], reverse=True)[:k]
        top_k_examples = [ex for ex, _ in top_k]
        return top_k_examples

        
    def pull_examples_from_json(self) -> list[Tuple[str, str]]:
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)
        quality_examples = data["quality_examples"]

        examples = [item['input']  for item in quality_examples]
        return examples
    
