from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import pandas as pd
import torch

@dataclass
class PreTrainedModel:
    model_name: str
    data: pd.DataFrame()
    column_name: str
    limit: int = 20
    corpus: str = ""
    
    def __post_init__(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.corpus = self.data[self.column_name].loc[:self.limit]


    def get_embedding(self, text: str) -> List[float]:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embedding = self.model(**tokens).last_hidden_state
        return embedding.mean(axis=1).squeeze(0).tolist()


    def process(self) -> List[Dict[str, List[float]]]:
        embeddings = []
        for row in self.corpus:
            embedding = self.get_embedding(row)
            index = {"id": row, "values": embedding}
            embeddings.append(index)
        return embeddings
            
        
        

    

    