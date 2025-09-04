# 📊 Data Analyst Agent ⚡️

A small Chainlit-based data analyst chat agent that answers questions about a synthetic business dataset, runs pandas / matplotlib code when needed, and returns outputs (stream) and inline PNG plots. 

## OpenAI Code Interpreter

- Released in **March 2023** (alpha for ChatGPT Plus).
- Integrated with **Assistant API**.
- **Assistant API** will be deprecated in 2026 ⚠️; use **Responses API** instead.
- **Responses API** creates an isolated container to execute Python code, with outputs such as images stored within the container.
- **Open-source alternatives** (e.g., Open Interpreter) are not actively developed; using **Responses API** is recommended. Integration with **Chainlit** has issues:
  1. [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) (v0.4.3) sometimes fails to install with Chainlit (v2.7.2) due to dependency collisions.
  2. [Code Interpreter API](https://github.com/shroominic/codeinterpreter-api) (v0.1.20) has issues with Azure OpenAI.

## Synthetic data description

- **synthetic\_business\_data.csv**

  - Synthetic, tabular business / transaction data for demo and analysis.

- **synthetic\_metadata.json**

  - JSON object describing columns and their human-readable descriptions.

## Usage

1. Create a `.env` with required Azure variables:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT_NAME`
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_API_VERSION`
2. Install dependencies:
   - `poetry install`
3. Run the app:
   - `python app.py`
4. In the chat, ask questions about the dataset. The assistant will run code for calculations/plots and return results.

## Notes

- Tool execution writes a Python script, runs it, and returns stdout/stderr plus any generated PNGs inline.
- File uploads are disabled in the chat flow.
- `app.legacy.py` is present in the repo but is considered legacy / not used by the code.
- If you plan to deploy or extend for Azure, follow Azure best practices, secure `.env` secrets 🔐, and review deployment guidance.

