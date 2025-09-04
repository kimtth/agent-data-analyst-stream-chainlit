import os, json, tempfile, asyncio, shutil
import pandas as pd
import chainlit as cl
from chainlit.input_widget import Select
from textwrap import dedent
from openai import AzureOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass, field


load_dotenv()

# Define tools for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run pandas/matplotlib code on df and return output plus PNG plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["code", "description"],
            },
        },
    }
]


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "synthetic_business_data.csv"
)
META_PATH = os.path.join(os.path.dirname(__file__), "data", "synthetic_metadata.json")


@dataclass
class AppState:
    df: pd.DataFrame | None = None
    metadata: dict | None = None
    provider: str = "Azure OpenAI"
    include_metadata: bool = True
    client: any = None
    history: list = field(default_factory=list)
    max_turns: int = 10
    data_filename: str = "synthetic_business_data.csv"
    meta_filename: str = "synthetic_metadata.json"
    data_path: str | None = None
    meta_path: str | None = None


def get_state() -> AppState:
    state = cl.user_session.get("state")
    if not state:
        state = AppState()
        cl.user_session.set("state", state)
    return state


# ---------- Data Load ----------
def load_data(force=False):
    state = get_state()
    if state.df is not None and not force:
        return
    state.data_path = DATA_PATH
    try:
        state.meta_path = META_PATH
    except FileNotFoundError:
        state.meta_path = None
    state.df = pd.read_csv(state.data_path)
    state.metadata = None
    if state.meta_path:
        with open(state.meta_path, "r", encoding="utf-8") as f:
            state.metadata = json.load(f)


# ---------- Settings ----------
def apply_settings(settings):
    state = get_state()
    mm = settings.get("metadata_mode", "With Metadata")
    state.include_metadata = mm == "With Metadata"
    if state.provider == "Azure OpenAI":
        # Prefer settings value, then env (supports .env.sample variable names)
        state.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )


# ---------- Prompt ----------
def build_system_prompt():
    state = get_state()
    head = state.df.head(5).to_markdown()
    if state.include_metadata and state.metadata:
        meta = "\n".join(
            f"{c['Column_name']}: {c['Description']['en']}"
            for c in state.metadata.get("columns", [])
        )
    else:
        meta = "(metadata disabled)"
    return dedent(
        f"""
        You are a data analyst. DataFrame 'df' is preloaded.
        Provide brief answers; use the tool for any calculation or plot.

        Metadata: {meta}
        Preview:
        {head}
        """
    ).strip()


# ---------- Tool Execution ----------
async def execute_python_tool(code: str, description: str) -> str:
    from pathlib import Path

    state = get_state()
    if state.df is None:
        load_data()
    tmp = Path(tempfile.mkdtemp(prefix="exec_"))
    data_path = tmp / "data.csv"
    state.df.to_csv(data_path, index=False)
    meta_obj = state.metadata if (state.include_metadata and state.metadata) else None
    script = tmp / "run.py"
    script.write_text(
        f"""
        import pandas as pd, json, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.read_csv(r"{data_path}")
        metadata = {json.dumps(meta_obj, ensure_ascii=False)}
        # user code
        {code.strip()}
        """,
        encoding="utf-8",
    )
    cmd = [os.getenv("PYTHON_EXECUTABLE", "python"), str(script)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(tmp),
        )
        try:
            so, se = await asyncio.wait_for(proc.communicate(), timeout=25)
        except asyncio.TimeoutError:
            proc.kill()
            shutil.rmtree(tmp, ignore_errors=True)
            return "Timeout."
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return f"Failed: {e}"
    out = so.decode() + (("\n[stderr]\n" + se.decode()) if se else "")
    for img in sorted(tmp.glob("*.png")):
        try:
            await cl.Image(
                name=img.name, content=img.read_bytes(), display="inline"
            ).send()
        except:
            pass
    if len(out) > 3000:
        out = out[:3000] + "...(truncated)"
    shutil.rmtree(tmp, ignore_errors=True)
    return out.strip()


async def handle_tool_calls(tool_calls):
    results = []
    for tc in tool_calls:
        if tc.function.name == "execute_python":
            args = json.loads(tc.function.arguments)
            code = args.get("code", "")
            desc = args.get("description", "")
            await cl.Message(content=f"Running: {desc}").send()
            try:
                output = await execute_python_tool(code, desc)
            except Exception as e:
                output = f"Error: {e}"
            results.append({"tool_call_id": tc.id, "role": "tool", "content": output})
    return results


# ---------- LLM ----------
async def llm_reply_with_tools(user_content: str) -> str:
    state = get_state()
    system = build_system_prompt()
    msgs = [{"role": "system", "content": system}]
    msgs.extend(state.history[-(2 * state.max_turns) :])
    msgs.append({"role": "user", "content": user_content})
    resp = state.client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=msgs,
        tools=TOOLS,
        tool_choice="auto",
    )
    am = resp.choices[0].message
    if am.content:
        await cl.Message(content=am.content).send()
    if am.tool_calls:
        tool_results = await handle_tool_calls(am.tool_calls)
        msgs.append(
            {
                "role": "assistant",
                "content": am.content or "",
                "tool_calls": am.tool_calls,
            }
        )
        msgs.extend(tool_results)
        final = state.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=msgs,
        )
        fc = final.choices[0].message.content
        if fc:
            await cl.Message(content=fc).send()
        return fc or (am.content or "")
    return am.content or ""


# ---------- History ----------
def record(role: str, content: str):
    state = get_state()
    state.history.append({"role": role, "content": content})
    if len(state.history) > 2 * state.max_turns + 4:
        del state.history[0:2]


# ---------- Chainlit Events ----------
@cl.on_chat_start
async def start():
    load_data(force=True)
    settings = await cl.ChatSettings(
        [
            Select(
                id="provider",
                label="Provider",
                values=["Azure OpenAI"],
                initial_index=0,
            ),
            Select(
                id="metadata_mode",
                label="Metadata",
                values=["With Metadata", "Without Metadata"],
                initial_index=0,
            ),
        ]
    ).send()
    apply_settings(settings)
    await cl.Message(content="Ready. Ask a question.").send()


@cl.on_settings_update
async def update(settings):
    apply_settings(settings)
    state = get_state()
    await cl.Message(
        content=f"Settings updated: provider={state.provider} metadata={state.include_metadata}"
    ).send()


@cl.on_message
async def handle(message: cl.Message):
    # Block any uploaded files/elements
    if getattr(message, "elements", None):
        await cl.Message(
            content="File uploads are disabled. Please paste text or code instead."
        ).send()
        return

    text = message.content.strip()

    if text.lower() in {"reload", "reload dataset", "reload context"}:
        load_data(force=True)
        await cl.Message(content="Dataset reloaded.").send()
        return

    # Use tool calling for regular messages
    record("user", text)
    try:
        reply = await llm_reply_with_tools(text)
        if reply:
            record("assistant", reply)
    except Exception as e:
        await cl.Message(content=f"❌ Error processing request: {str(e)}").send()


@cl.on_chat_end
async def _end():
    print("Chat ended.")
    pass


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
