from ollama import Client

URL = "http://localhost:11434" #url used for local ollama server
"""
This file is the Ollama client, it is the generic LLM that is built into the web application. It can use any model that is available in the Ollama server. The default model is llama3.2, but it can be changed by passing a different model name to the constructor of the OllamaClient class.
Before starting the project make sure to install the ollama package and run the ollama server. You can find the instructions to do so in the README.md file.
that in the Readme.md file.

"""

"""
====== Code Architecture ======
OllamaClient
 - client is the ollama client that is used to communicate with the ollama server
 - model_name is the name of the model that is used to generate text
 - generate method is implemented to generate text from the ollama client. All the of the parameters are defaulted except the prompt. 
 """
class OllamaClient:
    def __init__(self, model_name: str = "llama3.2"):
        self.client = Client(host=URL)
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 100, temp: float = 0, top_p: float = 1) -> str:

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p
            }
        )
        return response.response

