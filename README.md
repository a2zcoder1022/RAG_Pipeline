# LLM Learning Lab

This project is a small web app for learning about large language models. It has a React front end, a few Python experiments, and a local Ollama client.

## Front end

The React app is in `frontend/`. It currently includes:

- A searchable library of short guides about tokens, embeddings, attention, Transformers, prompt engineering, RAG, fine-tuning, inference, and evaluation.
- A separate page for each topic.
- A Prompt Lab for trying zero-shot and few-shot prompts, editing examples, previewing the prompt, and comparing the model's answer with an expected answer.

The topic guides and Prompt Lab form use sample content. They do not load a knowledge base or connect to a real dataset.

## Prompt engineering

Prompt engineering is writing instructions, context, and examples to guide a model's response. It does not update the model's weights.

- **Zero-shot:** Give the model instructions and a task, without examples.
- **One-shot:** Give it one input-and-answer example.
- **Few-shot:** Give it a small number of examples before the task.

The Prompt Lab uses sample support-ticket categories: Billing, Technical Support, and Account Access. The sample dataset is in `LLM/PromptEngineering/support_ticket_dataset.json`. It keeps few-shot examples separate from evaluation examples so the same test messages can be used to compare both approaches.

## Ollama API connection

The Prompt Lab is set up to send a request to `POST /api/prompt-engineering/run`. During local development, Vite forwards `/api` requests to FastAPI at `http://127.0.0.1:8000`.

The front end sends a JSON body with `mode`, `model`, `instruction`, `input`, `examples`, `temperature`, and `top_p`. Few-shot examples contain `input` and `expected_output`. It expects a JSON response with a string `response` field.

The current FastAPI app only has a `/hello` endpoint, so the prompt endpoint still needs to be added on the Python side. No Python files were changed for the Prompt Lab.

## Run the front end

From the project directory:

```powershell
cd frontend
npm install
npm run dev
```

Open the local address printed by Vite. To use the Prompt Lab, start the FastAPI server too and add the endpoint described above.
