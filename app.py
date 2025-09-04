import os
import chainlit as cl
from openai import AzureOpenAI
from loguru import logger
from dotenv import load_dotenv
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCompletedEvent,
    ResponseErrorEvent,
)
from typing import cast
import requests

load_dotenv()  # take environment variables from .env.

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "synthetic_business_data.csv"
)
META_PATH = os.path.join(os.path.dirname(__file__), "data", "synthetic_metadata.json")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
# Logging (stdout + file)
logger.remove()
logger.add(lambda m: print(m, end=""), level=os.getenv("LOG_LEVEL", "INFO"))
_logs_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(_logs_dir, exist_ok=True)
_log_file = os.getenv("LOG_FILE", os.path.join(_logs_dir, "app.log"))
logger.add(
    _log_file,
    level=os.getenv("LOG_LEVEL", "DEBUG"),  # INFO or DEBUG
    rotation="5 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
)


def system_instruction():
    return (
        "You are a data analyst. Use the Python code interpreter with the attached CSV for answers, KPIs, plots. "
        "If the user provided metadata, use it to understand column meanings and types. "
        "Give brief explanation then results. Compact tables. Reply in Japanese if user uses Japanese. "
        "When you create a plot, save it to /mnt/data with a clear filename and mention that filename."
        "Ensure that plot annotations use English only."
    )


def upload_files(include_meta: bool):
    logger.debug(f"Uploading files include_meta={include_meta}")
    f_csv = client.files.create(file=open(DATA_PATH, "rb"), purpose="assistants")
    f_json = None
    if include_meta:
        f_json = client.files.create(file=open(META_PATH, "rb"), purpose="assistants")
    return f_csv, f_json


@cl.on_chat_start
async def start():
    # First: Ask whether to load the bundled sample dataset
    ds_choice = await cl.AskActionMessage(
        content="Load the sample dataset file?",
        actions=[
            cl.Action(
                name="load_yes", payload={"load": True}, label="Yes (load sample)"
            ),
            cl.Action(
                name="load_no", payload={"load": False}, label="No (start empty)"
            ),
        ],
        timeout=6000,
    ).send()
    load_dataset = (ds_choice or {}).get("payload", {}).get("load", True)

    data_file_id = None
    include_meta = False

    if load_dataset:
        # Ask for metadata usage only if dataset will be available
        meta_sel = await cl.AskActionMessage(
            content="Select metadata usage:",
            actions=[
                cl.Action(
                    name="with_meta", payload={"mode": "with"}, label="With metadata"
                ),
                cl.Action(
                    name="without_meta",
                    payload={"mode": "without"},
                    label="Without metadata",
                ),
            ],
            timeout=6000,
        ).send()
        mode = (meta_sel or {}).get("payload", {}).get("mode", "with")
        include_meta = mode == "with"

        # Upload dataset (and optionally could upload metadata later if needed)
        f_csv, f_json = upload_files(include_meta)
        await cl.Message(
            content="Dataset uploaded. You can now ask data questions."
        ).send()

        data_file_id = f_csv.id
        meta_data_file_id = f_json.id if f_json else None
        cl.user_session.set("data_file_id", data_file_id)
        cl.user_session.set("meta_data_file_id", meta_data_file_id)
    else:
        cl.user_session.set("data_file_id", None)
        cl.user_session.set("meta_data_file_id", None)

    sys_msg = {
        "role": "system",
        "content": system_instruction()
        + (
            "\n(Note: No dataset loaded. You may ask the user to upload one or provide raw data.)"
            if not load_dataset
            else ""
        ),
    }
    cl.user_session.set("messages", [sys_msg])
    cl.user_session.set("include_meta", include_meta)
    await cl.Message(
        content=(
            f"Ready. Dataset: {'Loaded' if load_dataset else 'Not loaded'}."
            + (
                f" Metadata: {'With' if include_meta else 'Without'}."
                if load_dataset
                else ""
            )
            + " Ask your data question."
        )
    ).send()
    logger.debug(f"Chat start load_dataset={load_dataset}")


def run_response_sync(messages):
    """Original (non-streaming) implementation kept for fallback."""
    file_ids = [
        fid
        for fid in (
            cl.user_session.get("data_file_id"),
            cl.user_session.get("meta_data_file_id"),
        )
        if fid
    ]
    sys_content = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo_msgs = [
        {
            "role": m["role"],
            "content": [
                {
                    "type": ("input_text" if m["role"] == "user" else "output_text"),
                    "text": m["content"],
                }
            ],
        }
        for m in messages
        if m["role"] != "system"
    ]
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    try:
        logger.info(
            f"[sync] Request model={model} files={len(file_ids)} msgs={len(convo_msgs)}"
        )
        resp = client.responses.create(
            model=model,
            instructions=sys_content,
            input=convo_msgs,
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto", "file_ids": file_ids},
                }
            ],
            # Remove stream=True for sync version
        )
        logger.info(
            f"[sync] Response id={getattr(resp,'id','n/a')} blocks={len(resp.output or [])}"
        )
        return resp
    except Exception as e:
        logger.error(f"[sync] API error: {e}")
        return {"error": str(e)}


def extract_text_parts(resp):
    parts = []
    for block in getattr(resp, "output", []) or []:
        if getattr(block, "type", "") == "message":
            for part in getattr(block, "content", []) or []:
                if getattr(part, "type", "") == "output_text":
                    txt = getattr(part, "text", "")
                    if txt:
                        parts.append(txt)
    logger.debug(f"Extracted {''.join(parts)}")
    return parts


def extract_usage_summary(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return ""
    try:
        it = u.get("input_tokens")
        ot = u.get("output_tokens")
        tt = u.get("total_tokens")
        logger.debug(f"Usage: in={it} out={ot} total={tt}")
        return f"(tokens in/out/total: {it}/{ot}/{tt})"
    except:
        return ""


# --- Added helpers for container file citations ---
def _iter_container_citations(resp):
    for block in getattr(resp, "output", []) or []:
        if (
            getattr(block, "type", "") == "message"
            and getattr(block, "role", "") == "assistant"
        ):
            for part in getattr(block, "content", []) or []:
                if getattr(part, "type", "") == "output_text":
                    for ann in getattr(part, "annotations", []) or []:
                        if getattr(ann, "type", "") == "container_file_citation":
                            yield (
                                getattr(ann, "container_id", None),
                                getattr(ann, "file_id", None),
                                getattr(ann, "filename", "output.png"),
                            )


def _download_container_file(container_id: str, file_id: str) -> bytes:
    base = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not (base and api_version):
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_VERSION")
    
    url = f"{base}/openai/v1/containers/{container_id}/files/{file_id}/content?api-version=preview"
    r = requests.get(
        url,
        headers={"api-key": os.getenv("AZURE_OPENAI_API_KEY")},
        stream=True,
        timeout=60,
    )
    r.raise_for_status()
    return b"".join(r.iter_content(8192))
# --- end helpers ---


USE_STREAMING = os.getenv("USE_STREAMING", "1").lower() in ("1", "true", "yes")


@cl.on_message
async def on_message(msg: cl.Message):
    logger.info("User message")
    messages = cl.user_session.get("messages")
    if not messages:
        await cl.Message(content="Session not initialized.").send()
        return
    messages.append({"role": "user", "content": msg.content})

    if USE_STREAMING:
        await _handle_streaming(messages)
    else:
        await cl.Message(content="Running analysis... (non-streaming)").send()
        resp = run_response_sync(messages)

        logger.debug(f"full response: {resp}  ")

        if isinstance(resp, dict) and resp.get("error"):
            await cl.Message(content=f"Error: {resp['error']}").send()
            return

        status = getattr(resp, "status", None)
        if status and status != "completed":
            await cl.Message(content=f"Response status: {status} (partial)").send()
        image_elements = _gather_image_elements(resp)
        usage_note = extract_usage_summary(resp)
        final_text = (
            "\n".join(t for t in extract_text_parts(resp) if t).strip() or "(no text)"
        )
        if usage_note:
            final_text += f"\n{usage_note}"
        await cl.Message(content=final_text, elements=image_elements or None).send()
        messages.append({"role": "assistant", "content": final_text})


# --- Simplified Streaming handler ---
async def _handle_streaming(messages):
    live_msg = cl.Message(content="Now analyzing...")
    await live_msg.send()
    text_chunks = []
    final_resp = None

    try:
        # Use proper streaming context manager API
        with client.responses.stream(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            instructions=next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            ),
            input=[
                {
                    "role": m["role"],
                    "content": [
                        {
                            "type": (
                                "input_text" if m["role"] == "user" else "output_text"
                            ),
                            "text": m["content"],
                        }
                    ],
                }
                for m in messages
                if m["role"] != "system"
            ]
            or [
                {"role": "user", "content": [{"type": "input_text", "text": "Ready."}]}
            ],
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": [
                            fid
                            for fid in (
                                cl.user_session.get("data_file_id"),
                                cl.user_session.get("meta_data_file_id"),
                            )
                            if fid
                        ],
                    },
                }
            ],
        ) as stream:
            code_snip = None  # track current code block
            for event in stream:
                event = cast(ResponseStreamEvent, event)

                match event:
                    # ResponseTextDeltaEvent = "response.output_text.delta"
                    case ResponseTextDeltaEvent(delta=delta) if delta:
                        text_chunks.append(delta)
                        await live_msg.stream_token(delta)
                    case ResponseCodeInterpreterCallCodeDeltaEvent(
                        delta=delta
                    ) if delta:
                        if code_snip is None:  # new block starting
                            # first chunk → start code fence
                            code_snip = ""
                            text_chunks.append("\n```python\n")
                            await live_msg.stream_token("\n```python\n")
                        code_snip += delta
                        text_chunks.append(delta)
                        await live_msg.stream_token(delta)
                    case ResponseCompletedEvent():
                        # et = "response.completed"
                        final_resp = getattr(event, "response", None)
                    case ResponseErrorEvent():
                        # et = "response.error"
                        err_obj = getattr(event, "error", None)
                        if err_obj:
                            live_msg.content = f"Error: {err_obj}"
                            await live_msg.update()
                    # Completed code segment
                    case ResponseCodeInterpreterCallCompletedEvent():
                        if code_snip is not None:
                            # close the code fence
                            text_chunks.append("\n```\n")
                            await live_msg.stream_token("\n```\n")
                            code_snip = None  # reset for next snippet

        if final_resp:
            logger.debug("[stream] final model response received")
            usage_note = extract_usage_summary(final_resp)
            image_elements = _gather_image_elements(final_resp)
        else:
            logger.warning("[stream] No final model response (only deltas collected)")
            usage_note = ""
            image_elements = []

        full_text = "".join(text_chunks).strip()
        if not full_text and final_resp:
            full_text = "\n".join(extract_text_parts(final_resp)).strip() or "(no text)"
        elif not full_text:
            full_text = "(no text)"

        if usage_note:
            full_text += f"\n{usage_note}"

        live_msg.content = full_text
        live_msg.elements = image_elements if image_elements else []
        # Streaming is done, so update the message with final content
        await live_msg.update()
        messages.append({"role": "assistant", "content": full_text})

    except Exception as e:
        logger.error(f"[stream] failure: {e}")
        logger.info("[stream] Falling back to non-streaming mode")


def _gather_image_elements(resp):
    """Reuse existing logic to download images & cited files after final response."""
    image_elements = []

    # Container citations - keep existing logic
    cited_pairs = set()
    for cid, fid, fname in _iter_container_citations(resp):
        if not cid or not fid:
            continue
        key = (cid, fid)
        if key in cited_pairs:
            continue
        try:
            img_bytes = _download_container_file(cid, fid)
            image_elements.append(
                cl.Image(name=fname, content=img_bytes, display="inline")
            )
            cited_pairs.add(key)
            logger.debug(f"Downloaded cited container file {fname} ({fid})")
        except Exception as e:
            logger.warning(f"citation fetch failed {fid}: {e}")

    return image_elements


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
