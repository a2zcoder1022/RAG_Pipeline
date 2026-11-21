from LLM.ollama import OllamaClient


client = OllamaClient(model_name="llama3.2")


response = client.generate(prompt="What is a constant?", max_tokens=50, temp=1, top_p=0.9)

print(response)