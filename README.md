# LLM Prompt Engineering Lab

This project is a small React app for testing prompts with a local Ollama model.

## Front end

The React app is in `frontend/`. It has one page, Prompt Lab, for testing zero-shot, one-shot, and few-shot prompts against support-ticket examples from the JSON dataset. It shows model responses and compares them with expected labels.

## Prompt engineering

Prompt engineering is writing instructions, context, and examples to guide a model's response. It does not update the model's weights.

- **Zero-shot:** Give the model instructions and a task, without examples.
- **One-shot:** Give it one input-and-answer example.
- **Few-shot:** Give it a small number of examples before the task.

Prompt Lab uses the support-ticket categories Billing, IT, and Fraud. The dataset is in `LLM/PromptEngineering/support_ticket_dataset.json`. It keeps the original generic examples under `generic_examples`, while Prompt Lab uses the 75 `quality_examples` for one-shot and few-shot requests (25 per label). Its 40 held-out evaluation examples are balanced as evenly as possible across the three labels (14 Billing, 13 IT, and 13 Fraud).

## Back end

The API is in `API/api.py`. It exposes `GET /hello`, `GET /zero_shot_prompt/ollama`, `POST /one_shot_prompt/ollama`, and `POST /few_shot_prompt/ollama`. The zero-shot endpoint accepts a `prompt` query parameter. The one-shot endpoint accepts an instruction, one labeled example, and the new prompt. The few-shot endpoint accepts an instruction, a list of labeled examples, and the new prompt. All generation endpoints accept an optional `model_name` query parameter. Prompt Lab offers Llama 3.2, Qwen3 8B, Gemma3 4B, Mistral 7B, and Phi-4. Random example choices are reused for the same evaluation settings across models. **Run all models** executes the five model runs concurrently, processing prompts sequentially within each model. It shows progress by completed model and displays a final comparison table with overall accuracy, correct and incorrect totals, and IDs for incorrectly classified prompts. Individual prompt responses are not displayed.

`LLM/ollama.py` contains the `OllamaClient` wrapper. It connects to the local Ollama service at `http://localhost:11434`, uses `llama3.2` by default, and returns the generated response. Prompt Lab passes its selected model to each generation endpoint.

### Install and run Ollama

1. Install Ollama for your operating system from [ollama.com/download](https://ollama.com/download).
2. Open the Ollama app. In PowerShell, download whichever models you want to test:

   ```powershell
   ollama pull llama3.2
   ollama pull qwen3:8b
   ollama pull gemma3:4b
   ollama pull mistral:7b
   ollama pull phi4
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

### Run the Python experiment

Start the Ollama app, then run this from the project directory using a Python environment with PyTorch and Transformers installed:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'C:\Users\aaron\torch_test'
uv run --no-sync python -m Experiment.experiment --mode few_shot --selection top_k --k 3
```

Use `--selection random` to sample examples instead, or `--mode zero_shot` to skip examples and embeddings. The script runs models sequentially and writes summaries and per-prompt responses to `Experiment/experiment_results.json`. Choose a `torch_test` path that exists on your machine.

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
