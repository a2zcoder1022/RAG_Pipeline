# LLM Prompt Engineering Lab

This project is a small React app for testing prompts with a local Ollama model.

## Front end

The React app is in `frontend/`. It has one page, Prompt Lab, for testing zero-shot, one-shot, and few-shot prompts against support-ticket examples from the JSON dataset. It shows model responses and compares them with expected labels.

## Prompt engineering

Prompt engineering is writing instructions, context, and examples to guide a model's response. It does not update the model's weights.

- **Zero-shot:** Give the model instructions and a task, without examples.
- **One-shot:** Give it one input-and-answer example.
- **Few-shot:** Give it a small number of examples before the task.

Prompt Lab uses the support-ticket categories Billing, IT, and Fraud. The dataset is in `LLM/PromptEngineering/support_ticket_dataset.json`. Its few-shot examples are separate from the held-out evaluation examples.

## Back end

The API is in `API/api.py`. It exposes `GET /hello`, `GET /zero_shot_prompt/ollama`, `POST /one_shot_prompt/ollama`, and `POST /few_shot_prompt/ollama`. The zero-shot endpoint accepts a `prompt` query parameter. The one-shot endpoint accepts an instruction, one labeled example, and the new prompt. The few-shot endpoint accepts an instruction, a list of labeled examples, and the new prompt. Prompt Lab randomly selects one training example from each category for every few-shot ticket request and shows the selected examples in that result row. The endpoints send the assembled prompt to the Ollama client and return the generated text as JSON.

`LLM/ollama.py` contains the `OllamaClient` wrapper. It connects to the local Ollama service at `http://localhost:11434`, uses `llama3.2` by default, and returns the generated response. The API currently selects `llama3.2`.

### Install and run Ollama

1. Install Ollama for your operating system from [ollama.com/download](https://ollama.com/download).
2. Open the Ollama app. In PowerShell, download the model used by the API:

   ```powershell
   ollama pull llama3.2
   ```

3. Optionally, try the model in the terminal:

   ```powershell
   ollama run llama3.2
   ```

Ollama normally starts its local service when the app is running. If it is not running, start it in a separate terminal with `ollama serve`.

### Run the API

From the project directory, install the Python project dependencies and start FastAPI:

```powershell
uv sync
uv run uvicorn API.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to view and try the API endpoints.

## Run the front end

From the project directory:

```powershell
cd frontend
npm install
npm run dev
```

Open the local address printed by Vite. To run prompts, start the FastAPI server from the project directory:

```powershell
uv run uvicorn API.api:app --reload
```
