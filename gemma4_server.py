# -*- coding: utf-8 -*-
"""
Gemma 4 (12B) Unsloth | GDrive Binary with Auto LD_LIBRARY_PATH | Colab T4
"""
from __future__ import annotations

import asyncio
import atexit
import collections
import hmac
import importlib.util
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from typing import Any, AsyncIterator, Optional

# =============================================================================== dependencies & imports
def ensure_dependencies(need_ngrok: bool) -> None:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "httpx": "httpx",
        "huggingface_hub": "huggingface_hub",
        "gdown": "gdown"
    }
    if need_ngrok:
        required["pyngrok"] = "pyngrok"
    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()

def _env_int(name: str, default: int) -> int:
    try: return int(_env(name, str(default)))
    except ValueError: return default

def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")

def get_secret(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v: return v
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(name)
        if val: return str(val).strip()
    except Exception: pass
    return ""

ensure_dependencies(bool(get_secret("NGROK_AUTH_TOKEN")))

import gdown
import httpx
import uvicorn
from contextlib import asynccontextmanager
from starlette.requests import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# =============================================================================== config
def _pick_dir(env_name: str, colab_path: str, fallback: str) -> str:
    p = os.environ.get(env_name, "").strip()
    if p: return p
    if os.path.isdir("/content"): return colab_path
    return fallback

GDRIVE_BIN_URL = _env("GDRIVE_BIN_URL", "https://drive.google.com/file/d/1ARn96KgjNyZsqjsEn-VZD4IQFH5QOioI/view?usp=drivesdk")

MODEL_REPO = _env("MODEL_REPO", "unsloth/gemma-4-12B-it-qat-GGUF")
GGUF_FILE = _env("GGUF_FILE", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf") 
SERVED_NAME = _env("SERVED_NAME", "gemma-4-12b-agent")

LLAMA_EXTRA_ARGS = _env("LLAMA_EXTRA_ARGS", "-fa off")

BACKEND_PORT = _env_int("BACKEND_PORT", 8000)
GATEWAY_PORT = _env_int("GATEWAY_PORT", 8001)
GATEWAY_HOST = _env("GATEWAY_HOST", "127.0.0.1")

CTX_SIZE = _env_int("CTX_SIZE", 32768) 
PARALLEL = max(1, _env_int("PARALLEL", 1))
KV_CACHE_TYPE = _env("KV_CACHE_TYPE", "f16").lower()
TENSOR_SPLIT = _env("TENSOR_SPLIT", "auto")
LLAMA_VERBOSE = _env_bool("LLAMA_VERBOSE", False)

TOOL_MODE = _env("TOOL_MODE", "buffered").lower()
ENABLE_THINKING = _env("ENABLE_THINKING", "")
DEFAULT_MAX_TOKENS = _env_int("DEFAULT_MAX_TOKENS", 4096)
RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 120)
KEEPALIVE_SEC = _env_int("KEEPALIVE_SEC", 15)
QUEUE_TIMEOUT = _env_int("QUEUE_TIMEOUT", 600)
UPSTREAM_READ_TIMEOUT = _env_int("UPSTREAM_READ_TIMEOUT", 900)
MAX_BODY_BYTES = _env_int("MAX_BODY_MB", 32) * 1024 * 1024
SELFTEST = _env_bool("SELFTEST", True)

HF_TOKEN = get_secret("HF_TOKEN")
NGROK_AUTH_TOKEN = get_secret("NGROK_AUTH_TOKEN")
NGROK_DOMAIN = _env("NGROK_DOMAIN", "")
API_KEY = get_secret("API_KEY")
API_KEY_GENERATED = False
if not API_KEY:
    API_KEY = "sk-" + secrets.token_urlsafe(24)
    API_KEY_GENERATED = True

DOWNLOAD_DIR = _pick_dir("DOWNLOAD_DIR", "/content/models", "/tmp/models")
LLAMA_CPP_DIR = _pick_dir("LLAMA_CPP_DIR", "/content/llama.cpp", "/tmp/llama.cpp")
CLOUDFLARED_BIN = _env("CLOUDFLARED_BIN", "/tmp/cloudflared")
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

os.environ.setdefault("HF_HOME", os.path.join(DOWNLOAD_DIR, ".hf_home"))
START_TS = int(time.time())

# =============================================================================== logging
logger = logging.getLogger("gemma4")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

# =============================================================================== tool-call parser
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=\s*([^>\s]+)\s*>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=\s*([^>\s]+)\s*>(.*?)</parameter>", re.S)

def strip_think(text: str) -> str:
    if "</think>" not in text: return text
    text = _THINK_RE.sub("", text)
    if "</think>" in text: text = text.split("</think>")[-1]
    return text

def _schema_map(tools: Optional[list]) -> dict:
    out: dict = {}
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            out[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return out

def _trim_one_newline(v: str) -> str:
    if v.startswith("\r\n"): v = v[2:]
    elif v.startswith("\n"): v = v[1:]
    if v.endswith("\r\n"): v = v[:-2]
    elif v.endswith("\n"): v = v[:-1]
    return v

def _coerce(raw: str, schema: Any) -> Any:
    types = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(types, str): types = [types]
    if types:
        if types == ["string"]: return raw
        s = raw.strip()
        if "boolean" in types and s.lower() in ("true", "false"): return s.lower() == "true"
        try: val = json.loads(s)
        except Exception: return raw
        if "string" in types and not isinstance(val, (dict, list)) and val is not None: return raw
        return val
    s = raw.strip()
    if (s and s[0] in "{[") or s in ("true", "false", "null"):
        try: return json.loads(s)
        except Exception: pass
    return raw

def _make_call(name: str, args: Any) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)},
    }

def parse_tool_calls(content: Optional[str], tools: Optional[list] = None) -> tuple[str, list]:
    if not content: return "", []
    text = strip_think(content)
    if "<tool_call>" not in text and "<function=" not in text and "```json" not in text:
        return text.strip(), []

    schemas = _schema_map(tools)
    blocks = _BLOCK_RE.findall(text) or ([text] if "<function=" in text else [])
    calls: list = []
    
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.S)
    for jb in json_blocks:
        try:
            obj = json.loads(jb)
            if isinstance(obj, dict) and "name" in obj:
                calls.append(_make_call(obj["name"], obj.get("arguments", obj.get("parameters", {}))))
        except Exception: pass

    for block in blocks:
        funcs = list(_FUNC_RE.finditer(block))
        if funcs:
            for m in funcs:
                name = m.group(1).strip()
                if schemas and name not in schemas: continue
                props = schemas.get(name, {})
                args = {}
                for pm in _PARAM_RE.finditer(m.group(2)):
                    pname = pm.group(1).strip()
                    args[pname] = _coerce(_trim_one_newline(pm.group(2)), props.get(pname))
                calls.append(_make_call(name, args))
        elif block.lstrip().startswith("{"):
            try: obj = json.loads(block)
            except Exception: continue
            name = obj.get("name") if isinstance(obj, dict) else None
            if not name or (schemas and name not in schemas): continue
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            calls.append(_make_call(name, args if isinstance(args, dict) else {}))

    if not calls: return text.strip(), []
    clean_text = text.split("<tool_call>")[0].split("<function=")[0].split("```json")[0].strip()
    return clean_text, calls

# =============================================================================== helpers
class GatewayError(Exception):
    def __init__(self, status: int, message: str, etype: str = "server_error"):
        super().__init__(message)
        self.status, self.message, self.etype = status, message, etype

class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict = {}

    def allow(self, key: str, now: Optional[float] = None) -> tuple[bool, int]:
        if self.per_min <= 0: return True, 0
        now = time.monotonic() if now is None else now
        dq = self.hits.setdefault(key, collections.deque())
        while dq and now - dq[0] > 60: dq.popleft()
        if len(dq) >= self.per_min: return False, int(60 - (now - dq[0])) + 1
        dq.append(now)
        if len(self.hits) > 5000:
            self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] <= 60}
        return True, 0

def err_body(message: str, etype: str = "server_error", code: Optional[str] = None) -> dict:
    return {"error": {"message": message, "type": etype, "code": code}}

def extract_error_message(raw: bytes, status: int) -> str:
    try:
        obj = json.loads(raw)
        e = obj.get("error", obj)
        if isinstance(e, dict): return str(e.get("message") or e)[:500]
        return str(e)[:500]
    except Exception:
        return (raw.decode("utf-8", "replace")[:300] or f"upstream HTTP {status}")

def describe_exc(e: BaseException) -> GatewayError:
    name = type(e).__name__
    if "Timeout" in name: return GatewayError(504, f"Upstream timeout ({name})", "timeout")
    return GatewayError(502, f"Upstream error: {name}: {e}", "server_error")

SSE_KEEPALIVE = b": keep-alive\n\n"
SSE_DONE = b"data: [DONE]\n\n"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_END = object()

def sse_error(message: str, etype: str = "server_error") -> bytes:
    return b"data: " + json.dumps(err_body(message, etype), ensure_ascii=False).encode() + b"\n\n"

def sse_chunk(cid: str, created: int, delta: dict, finish: Optional[str] = None) -> bytes:
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": SERVED_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

async def ka_ticks(task: asyncio.Future, interval: Optional[float], max_wait: Optional[float] = None) -> AsyncIterator[None]:
    interval = interval if interval and interval > 0 else 3600.0
    waited = 0.0
    while not task.done():
        await asyncio.wait({task}, timeout=interval)
        if not task.done():
            waited += interval
            yield
            if max_wait is not None and waited >= max_wait: return

async def _anext(it) -> Any:
    try: return await it.__anext__()
    except StopAsyncIteration: return _END

def normalize_tool_calls(calls: list) -> list:
    out = []
    for tc in calls:
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments", "{}")
        fn["arguments"] = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        out.append({"id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": fn})
    return out

def apply_tool_fallback(message: dict, finish: str, tools: Optional[list]) -> tuple[dict, str]:
    if message.get("tool_calls"):
        message["tool_calls"] = normalize_tool_calls(message["tool_calls"])
        return message, "tool_calls"
    text, calls = parse_tool_calls(message.get("content") or "", tools)
    if calls:
        message["content"] = text or None
        message["tool_calls"] = calls
        return message, "tool_calls"
    return message, finish

def synth_stream(resp: dict, tools: Optional[list], include_usage: bool) -> list[bytes]:
    cid = resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = resp.get("created") or int(time.time())
    choice = (resp.get("choices") or [{}])[0]
    msg, finish = apply_tool_fallback(dict(choice.get("message") or {}), choice.get("finish_reason") or "stop", tools)
    out = [sse_chunk(cid, created, {"role": "assistant", "content": ""})]
    if msg.get("reasoning_content"): out.append(sse_chunk(cid, created, {"reasoning_content": msg["reasoning_content"]}))
    if msg.get("content"): out.append(sse_chunk(cid, created, {"content": msg["content"]}))
    if msg.get("tool_calls"): out.append(sse_chunk(cid, created, {"tool_calls": [{"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])]}))
    out.append(sse_chunk(cid, created, {}, finish))
    if include_usage and resp.get("usage"):
        out.append(b"data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": SERVED_NAME, "choices": [], "usage": resp["usage"]}).encode() + b"\n\n")
    out.append(SSE_DONE)
    return out

def prepare_payload(data: dict) -> None:
    data["model"] = SERVED_NAME
    mt = data.get("max_tokens")
    if isinstance(mt, int) and 0 < mt <= 1:
        data["max_tokens"] = 16
    elif not (data.get("max_tokens") or data.get("max_completion_tokens") or data.get("n_predict")):
        data["max_tokens"] = DEFAULT_MAX_TOKENS
    if ENABLE_THINKING in ("0", "1") and "chat_template_kwargs" not in data:
        data["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING == "1"}

# =============================================================================== gateway streaming
async def _acquire(slots: asyncio.Semaphore, ka: Optional[float], max_wait: float):
    acq = asyncio.ensure_future(slots.acquire())
    try:
        async for _ in ka_ticks(acq, ka, max_wait): yield None
        if acq.done() and not acq.cancelled(): yield True
        else:
            acq.cancel()
            yield False
    except BaseException:
        if not acq.done(): acq.cancel()
        elif not acq.cancelled(): slots.release()
        raise

async def passthrough_stream(client, url: str, headers: dict, body: bytes, slots: asyncio.Semaphore,
                             ka: Optional[float] = None) -> AsyncIterator[bytes]:
    ka = KEEPALIVE_SEC if ka is None else ka
    held = False
    task = nxt = upstream = None
    t0 = time.time()
    try:
        async for state in _acquire(slots, ka, QUEUE_TIMEOUT):
            if state is None: yield SSE_KEEPALIVE
            else: held = state
        if not held:
            yield sse_error("Server busy: queue timeout", "server_error")
            yield SSE_DONE
            return
        req = client.build_request("POST", url, headers=headers, content=body)
        task = asyncio.ensure_future(client.send(req, stream=True))
        async for _ in ka_ticks(task, ka): yield SSE_KEEPALIVE
        upstream = task.result()
        if upstream.status_code >= 400:
            raw = await upstream.aread()
            yield sse_error(extract_error_message(raw, upstream.status_code), "invalid_request_error")
            yield SSE_DONE
            return
        it = upstream.aiter_raw().__aiter__()
        nxt = asyncio.ensure_future(_anext(it))
        while True:
            done, _ = await asyncio.wait({nxt}, timeout=ka if ka and ka > 0 else 3600)
            if not done:
                yield SSE_KEEPALIVE
                continue
            chunk = nxt.result()
            if chunk is _END: break
            yield chunk
            nxt = asyncio.ensure_future(_anext(it))
    except asyncio.CancelledError: raise
    except Exception as e:
        ge = describe_exc(e)
        logger.error("stream failed: %s", ge.message)
        yield sse_error(ge.message, ge.etype)
        yield SSE_DONE
    finally:
        for t in (nxt, task):
            if t is not None and not t.done(): t.cancel()
        if upstream is not None:
            try: await upstream.aclose()
            except Exception: pass
        if held: slots.release()
        logger.info("stream finished in %.1fs", time.time() - t0)

async def buffered_tool_stream(client, url: str, headers: dict, body: bytes, slots: asyncio.Semaphore,
                               tools: list, include_usage: bool, ka: Optional[float] = None) -> AsyncIterator[bytes]:
    ka = KEEPALIVE_SEC if ka is None else ka
    held = False
    task = None
    t0 = time.time()
    try:
        async for state in _acquire(slots, ka, QUEUE_TIMEOUT):
            if state is None: yield SSE_KEEPALIVE
            else: held = state
        if not held:
            yield sse_error("Server busy: queue timeout", "server_error")
            yield SSE_DONE
            return
        task = asyncio.ensure_future(client.request("POST", url, headers=headers, content=body))
        async for _ in ka_ticks(task, ka): yield SSE_KEEPALIVE
        resp = task.result()
        if resp.status_code != 200:
            yield sse_error(extract_error_message(resp.content, resp.status_code), "invalid_request_error")
            yield SSE_DONE
            return
        for chunk in synth_stream(resp.json(), tools, include_usage): yield chunk
    except asyncio.CancelledError: raise
    except Exception as e:
        ge = describe_exc(e)
        logger.error("buffered stream failed: %s", ge.message)
        yield sse_error(ge.message, ge.etype)
        yield SSE_DONE
    finally:
        if task is not None and not task.done(): task.cancel()
        if held: slots.release()
        logger.info("buffered tool request finished in %.1fs", time.time() - t0)

async def run_nonstream(client, method: str, url: str, headers: dict, body: bytes,
                        slots: Optional[asyncio.Semaphore], tools: Optional[list]) -> tuple[int, str, bytes]:
    held = False
    if slots is not None:
        try:
            await asyncio.wait_for(slots.acquire(), QUEUE_TIMEOUT)
            held = True
        except asyncio.TimeoutError:
            raise GatewayError(503, "Server busy: queue timeout", "server_error")
    try:
        try:
            resp = await client.request(method, url, headers=headers, content=body or None)
        except asyncio.CancelledError: raise
        except Exception as e: raise describe_exc(e)
        ctype = resp.headers.get("content-type", "application/json")
        content = resp.content
        if tools and resp.status_code == 200:
            try:
                data = json.loads(content)
                choice = data["choices"][0]
                choice["message"], choice["finish_reason"] = apply_tool_fallback(
                    choice.get("message") or {}, choice.get("finish_reason") or "stop", tools)
                content = json.dumps(data, ensure_ascii=False).encode()
            except Exception as e:
                logger.warning("tool post-processing skipped: %s", e)
        return resp.status_code, ctype, content
    finally:
        if held: slots.release()

# =============================================================================== gateway app
def create_app(backend: Optional["Backend"] = None):
    limiter = RateLimiter(RATE_LIMIT_PER_MIN)
    ready = lambda: backend is None or backend.ready.is_set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=float(UPSTREAM_READ_TIMEOUT), write=60.0, pool=None),
            limits=httpx.Limits(max_connections=64),
        )
        app.state.slots = asyncio.Semaphore(PARALLEL)
        yield
        await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False)

    def jerr(status: int, message: str, etype: str = "server_error", code: Optional[str] = None, headers: Optional[dict] = None):
        return JSONResponse(err_body(message, etype, code), status_code=status, headers=headers)

    def client_ip(request: Request) -> str:
        h = request.headers
        return (h.get("cf-connecting-ip") or h.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else "unknown"))

    def authorized(request: Request) -> bool:
        tok = request.headers.get("authorization", "")
        tok = tok[7:] if tok.lower().startswith("bearer ") else tok
        tok = tok.strip() or request.headers.get("x-api-key", "").strip()
        return hmac.compare_digest(tok.encode(), API_KEY.encode())

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend_ready": ready(), "model": SERVED_NAME}

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def api_handler(path: str, request: Request):
        clean = path.strip("/")
        if clean.startswith("v1/"): clean = clean[3:].strip("/")
        if clean == "health": return {"status": "ok", "backend_ready": ready(), "model": SERVED_NAME}

        ok, retry = limiter.allow(client_ip(request))
        if not ok: return jerr(429, "Rate limit exceeded", "rate_limit_error", headers={"Retry-After": str(retry)})
        if not authorized(request): return jerr(401, "Invalid API key", "authentication_error", "invalid_api_key")
        if ".." in clean or "//" in clean: return jerr(400, "Invalid path", "invalid_request_error")

        if request.method == "GET" and clean == "models":
            return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "created": START_TS, "owned_by": "llama.cpp"}]}
        if request.method == "GET" and clean.startswith("models/"):
            return {"id": SERVED_NAME, "object": "model", "created": START_TS, "owned_by": "llama.cpp"}

        if not ready(): return jerr(503, "Model is still loading, retry shortly", "server_error", headers={"Retry-After": "15"})

        url = f"http://127.0.0.1:{BACKEND_PORT}/v1/{clean}" + (f"?{request.url.query}" if request.url.query else "")
        headers = {"accept-encoding": "identity", "content-type": "application/json"}
        if request.headers.get("accept"): headers["accept"] = request.headers["accept"]

        body = b""
        data: Optional[dict] = None
        tools: Optional[list] = None
        stream = False
        include_usage = False
        if request.method == "POST":
            try: declared = int(request.headers.get("content-length") or 0)
            except ValueError: declared = 0
            if declared > MAX_BODY_BYTES: return jerr(413, "Request body too large", "invalid_request_error")
            body = await request.body()
            try:
                parsed = json.loads(body)
                data = parsed if isinstance(parsed, dict) else None
            except Exception: data = None
            if data is not None:
                prepare_payload(data)
                stream = bool(data.get("stream"))
                tools = data.get("tools") if clean == "chat/completions" and data.get("tools") else None
                include_usage = bool((data.get("stream_options") or {}).get("include_usage"))
                if tools and TOOL_MODE == "buffered":
                    data["stream"] = False
                    data.pop("stream_options", None)
                body = json.dumps(data, ensure_ascii=False).encode()
            logger.info("%s POST /%s stream=%s tools=%s", client_ip(request), clean, stream, bool(tools))

        client, slots = app.state.client, app.state.slots
        if request.method == "POST" and stream:
            gen = (buffered_tool_stream(client, url, headers, body, slots, tools, include_usage)
                   if tools and TOOL_MODE == "buffered"
                   else passthrough_stream(client, url, headers, body, slots))
            return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)

        t0 = time.time()
        try:
            status, ctype, content = await run_nonstream(
                client, request.method, url, headers, body, slots if request.method == "POST" else None, tools)
        except GatewayError as e: return jerr(e.status, e.message, e.etype)
        logger.info("request finished in %.1fs (HTTP %s)", time.time() - t0, status)
        return Response(content=content, status_code=status, headers={"content-type": ctype})

    return app

# =============================================================================== binary setup (Drive via gdown)
def detect_gpus() -> list[dict]:
    for query in ("name,memory.total,compute_cap", "name,memory.total"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
        except Exception: continue
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 2: gpus.append({"name": p[0], "mem_mb": int(float(p[1])), "cc": p[2] if len(p) > 2 else "7.5"})
        if gpus: return gpus
    return []

def extract_archive(archive_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    try:
        shutil.unpack_archive(archive_path, extract_dir)
        return
    except Exception:
        pass

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_dir)
    else:
        res = subprocess.run(["7z", "x", archive_path, f"-o{extract_dir}", "-y"], capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to unpack archive: {archive_path}")

def ensure_llama_server(gpus: list[dict], has_gpu: bool) -> str:
    env_bin = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin): return env_bin
    if shutil.which("llama-server"): return shutil.which("llama-server")  # type: ignore[return-value]

    bin_dir = os.path.join(LLAMA_CPP_DIR, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    # التحقق من وجود الملف محلياً
    for root, _, files in os.walk(bin_dir):
        if "llama-server" in files:
            cached_bin = os.path.join(root, "llama-server")
            if os.access(cached_bin, os.X_OK):
                logger.info("Using cached llama-server: %s", cached_bin)
                # ضبط الصلاحيات للمكتبات المرافقة
                for r, _, fs in os.walk(bin_dir):
                    for f in fs:
                        if f.endswith(".so") or ".so." in f or f == "llama-server":
                            try: os.chmod(os.path.join(r, f), 0o755)
                            except OSError: pass
                return cached_bin

    archive_path = os.path.join(LLAMA_CPP_DIR, "llama_archive.pkg")
    logger.info("Downloading pre-compiled llama-server archive from Google Drive...")
    
    output = gdown.download(url=GDRIVE_BIN_URL, output=archive_path, quiet=False, fuzzy=True)
    if not output or not os.path.isfile(archive_path):
        raise RuntimeError("Failed to download binary from Google Drive via gdown.")

    logger.info("Extracting archive (190MB)...")
    extract_archive(archive_path, bin_dir)
    
    if os.path.exists(archive_path):
        try: os.remove(archive_path)
        except OSError: pass

    # منح صلاحيات التشغيل الكاملة للملف التنفيذي والمكتبات المشتركة
    found_bin = None
    for root, _, files in os.walk(bin_dir):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".so") or ".so." in f or f == "llama-server":
                try: os.chmod(p, 0o755)
                except OSError: pass
            if f == "llama-server" and not found_bin:
                found_bin = p
            
    if not found_bin:
        raise RuntimeError("llama-server executable was not found inside the extracted Drive package.")
        
    logger.info("llama-server is ready: %s", found_bin)
    return found_bin

def download_model() -> str:
    global DOWNLOAD_DIR
    try: os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except OSError:
        DOWNLOAD_DIR = "/tmp/models"
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    local = os.path.join(DOWNLOAD_DIR, GGUF_FILE)
    if os.path.isfile(local) and os.path.getsize(local) > 1 << 30 and not _env_bool("FORCE_DOWNLOAD", False):
        logger.info("Model already present: %s", local)
        return local

    free_gb = shutil.disk_usage(DOWNLOAD_DIR).free / 1e9
    logger.info("Downloading %s/%s to %s (free disk: %.0f GB)", MODEL_REPO, GGUF_FILE, DOWNLOAD_DIR, free_gb)
    if free_gb < 8:
        logger.warning("Low disk space! Model might fail to download.")
    from huggingface_hub import hf_hub_download, list_repo_files

    kw: dict = {"repo_id": MODEL_REPO, "local_dir": DOWNLOAD_DIR}
    if HF_TOKEN: kw["token"] = HF_TOKEN
    try: return hf_hub_download(filename=GGUF_FILE, **kw)
    except Exception as e:
        logger.warning("Direct download failed (%s: %s). Searching repo...", type(e).__name__, e)
    
    m = re.search(r"(IQ\d_\w+|Q\d_K_\w+|Q\d_K|Q\d_\d|BF16|F16)", GGUF_FILE)
    if not m: raise RuntimeError(f"Cannot download {GGUF_FILE} from {MODEL_REPO}")
    files = sorted(f for f in list_repo_files(MODEL_REPO, token=HF_TOKEN or None)
                   if f.endswith(".gguf") and m.group(1) in f and "mmproj" not in f.lower())
    if not files: raise RuntimeError(f"No *{m.group(1)}*.gguf file found in {MODEL_REPO}")
    logger.info("Using %s", files[0])
    paths = [hf_hub_download(filename=f, **kw) for f in (files if "-of-" in files[0] else files[:1])]
    return paths[0]

def kill_stale_servers() -> None:
    if not os.path.isdir("/proc"): return
    killed = False
    needle = f"\0--port\0{BACKEND_PORT}\0"
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        if int(pid) == os.getpid(): continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f: cmd = f.read().decode("utf-8", "ignore")
        except OSError: continue
        if os.path.basename(cmd.split("\0")[0]) == "llama-server" and needle in "\0" + cmd:
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed = True
                logger.info("Killed stale llama-server (pid %s)", pid)
            except OSError: pass
    if killed: time.sleep(3)

# =============================================================================== backend manager
class Backend:
    def __init__(self, binary: str, model_path: str, gpus: list[dict], has_gpu: bool):
        self.binary, self.model_path, self.gpus, self.has_gpu = binary, model_path, gpus, has_gpu
        self.proc: Optional[subprocess.Popen] = None
        self.ready = threading.Event()
        self.gave_up = False
        self.tail: collections.deque = collections.deque(maxlen=200)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._env: dict = {}

    def command(self) -> list[str]:
        cmd = [self.binary, "-m", self.model_path, "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
               "-c", str(CTX_SIZE), "--jinja", "--alias", SERVED_NAME]
        
        if self.has_gpu:
            cmd += ["-ngl", "999"]
            if len(self.gpus) > 1:
                ts = TENSOR_SPLIT if TENSOR_SPLIT != "auto" else ",".join(str(g["mem_mb"]) for g in self.gpus)
                cmd += ["-ts", ts]
        else:
            cpu_threads = str(max(2, os.cpu_count() or 4))
            cmd += ["-ngl", "0", "--threads", cpu_threads]
            
        if PARALLEL > 1: cmd += ["-np", str(PARALLEL)]
        if KV_CACHE_TYPE not in ("", "f16"): cmd += ["-ctk", KV_CACHE_TYPE, "-ctv", KV_CACHE_TYPE]
        if LLAMA_EXTRA_ARGS: cmd += shlex.split(LLAMA_EXTRA_ARGS)
        
        # تجميع مسارات المكتبات المشتركة تلقائياً وربطها بـ LD_LIBRARY_PATH
        bin_dir = os.path.dirname(self.binary)
        so_dirs = {bin_dir}
        for root, _, files in os.walk(LLAMA_CPP_DIR):
            for f in files:
                if f.endswith(".so") or ".so." in f:
                    so_dirs.add(root)

        for cuda_path in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu"):
            if os.path.isdir(cuda_path):
                so_dirs.add(cuda_path)

        extra_ld = os.pathsep.join(so_dirs)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = extra_ld + (os.pathsep + existing_ld if existing_ld else "")
        
        os.environ["LD_LIBRARY_PATH"] = new_ld
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = new_ld
        self._env = env
            
        return cmd

    def tail_text(self, n: int = 60) -> str: return "\n".join(list(self.tail)[-n:])

    def _pump(self, proc: subprocess.Popen) -> None:
        interesting = re.compile(r"(error|fail|warn|listening|loaded|offload|cuda|ready|out of memory|unknown model)", re.I)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            self.tail.append(line)
            if LLAMA_VERBOSE or interesting.search(line): logger.info("[llama] %s", line[:220])

    def _terminate(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=10)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass

    def start(self) -> None:
        with self._lock:
            self._terminate()
            self.ready.clear()
            cmd = self.command()
            device_msg = "GPUs" if self.has_gpu else "CPU"
            logger.info(f"Starting llama-server (loading Gemma model onto {device_msg})...")
            logger.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         errors="replace", bufsize=1, start_new_session=True, env=self._env)
            threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self._wait_ready()

    def _wait_ready(self, timeout: int = 1500) -> None:
        url = f"http://127.0.0.1:{BACKEND_PORT}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                raise RuntimeError(f"llama-server exited (code {code}). Last log lines:\n{self.tail_text()}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        self.ready.set()
                        logger.info("llama-server is ready")
                        return
            except Exception: pass
            time.sleep(2)
        raise TimeoutError("llama-server did not become ready in time")

    def supervise(self) -> None:
        restarts: collections.deque = collections.deque()
        backoff = 5
        while not self._stop.is_set():
            time.sleep(5)
            p = self.proc
            if p is None or p.poll() is None or self._stop.is_set(): continue
            self.ready.clear()
            logger.error("llama-server exited (code %s). Last log lines:\n%s", p.returncode, self.tail_text(25))
            now = time.time()
            while restarts and now - restarts[0] > 1800: restarts.popleft()
            if len(restarts) >= 5:
                logger.error("llama-server crashed 5 times in 30 min - giving up.")
                self.gave_up = True
                return
            restarts.append(now)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            try: self.start()
            except Exception as e: logger.error("restart failed: %s", e)

    def stop(self) -> None:
        self._stop.set()
        self._terminate()

# =============================================================================== tunnel
class Tunnel:
    def __init__(self, port: int):
        self.port = port
        self.url: Optional[str] = None
        self.kind = ""
        self.proc: Optional[subprocess.Popen] = None
        self._ngrok_checked = 0.0

    def start(self) -> Optional[str]:
        if NGROK_AUTH_TOKEN and self._start_ngrok(): return self.url
        return self._start_cloudflared()

    def _start_ngrok(self) -> bool:
        try:
            from pyngrok import conf, ngrok
            conf.get_default().auth_token = NGROK_AUTH_TOKEN
            try:
                for t in ngrok.get_tunnels(): ngrok.disconnect(t.public_url)
            except Exception: pass
            opts = {"domain": NGROK_DOMAIN} if NGROK_DOMAIN else {}
            url = ngrok.connect(addr=f"127.0.0.1:{self.port}", proto="http", **opts).public_url
            self.url = "https://" + url[7:] if url.startswith("http://") else url
            self.kind = "ngrok"
            logger.info("ngrok tunnel established")
            return True
        except Exception as e:
            logger.warning("ngrok failed (%s: %s). Falling back to Cloudflare...", type(e).__name__, e)
            return False

    def _ensure_cloudflared(self) -> None:
        if os.path.isfile(CLOUDFLARED_BIN) and os.access(CLOUDFLARED_BIN, os.X_OK): return
        tmp = CLOUDFLARED_BIN + ".part"
        logger.info("Downloading cloudflared...")
        with urllib.request.urlopen(CLOUDFLARED_URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.chmod(tmp, 0o755)
        subprocess.run([tmp, "--version"], check=True, capture_output=True, timeout=30)
        os.replace(tmp, CLOUDFLARED_BIN)

    def _start_cloudflared(self) -> Optional[str]:
        self._ensure_cloudflared()
        for attempt in range(1, 4):
            proc = subprocess.Popen([CLOUDFLARED_BIN, "tunnel", "--no-autoupdate", "--protocol", "http2",
                                     "--url", f"http://127.0.0.1:{self.port}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            found = threading.Event()
            holder: list[str] = []

            def drain(p=proc, ev=found, out=holder):
                assert p.stdout is not None
                for line in p.stdout:
                    if not ev.is_set():
                        m = URL_RE.search(line)
                        if m:
                            out.append(m.group(0))
                            ev.set()

            threading.Thread(target=drain, daemon=True).start()
            if found.wait(60):
                self.proc, self.url, self.kind = proc, holder[0], "cloudflare"
                logger.info("Cloudflare tunnel established")
                return self.url
            logger.warning("cloudflared did not report a URL (attempt %d/3)", attempt)
            proc.kill()
        return None

    def dead(self) -> bool:
        if self.kind == "cloudflare": return self.proc is None or self.proc.poll() is not None
        if self.kind == "ngrok" and time.time() - self._ngrok_checked > 30:
            self._ngrok_checked = time.time()
            try:
                from pyngrok import ngrok
                return len(ngrok.get_tunnels()) == 0
            except Exception: return True
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None: self.proc.kill()
        if self.kind == "ngrok":
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except Exception: pass

# =============================================================================== self-test / banner
def _local_post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{GATEWAY_PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")

def selftest() -> None:
    kw = {"model": SERVED_NAME, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = _local_post("/v1/chat/completions", {**kw, "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                                                 "max_tokens": 128})
        logger.info("[selftest] chat OK -> %r", (r["choices"][0]["message"].get("content") or "")[:60])
    except Exception as e:
        logger.warning("[selftest] chat FAILED: %s", e)
        return
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    try:
        r = _local_post("/v1/chat/completions", {**kw, "tools": tools, "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]})
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls and calls[0]["function"]["name"] == "get_weather":
            logger.info("[selftest] tool calling OK -> get_weather(%s)", calls[0]["function"]["arguments"])
        else:
            logger.warning("[selftest] tool calling returned no tool_calls (content=%r)", (msg.get("content") or "")[:120])
    except Exception as e:
        logger.warning("[selftest] tool call test FAILED: %s", e)

def print_banner(url: str, has_gpu: bool) -> None:
    key = API_KEY if API_KEY_GENERATED else "(the API_KEY you configured)"
    mode = "GPU ACCELERATED" if has_gpu else "CPU ONLY (Slow mode)"
    print("\n" + "=" * 62, flush=True)
    print(f"GEMMA 4 (12B) SYSTEM IS LIVE [{mode}]", flush=True)
    print(f"Base URL : {url}/v1", flush=True)
    print(f"Model    : {SERVED_NAME}", flush=True)
    print(f"API key  : {key}", flush=True)
    if API_KEY_GENERATED:
        print("(generated for this session - set API_KEY in Colab Secrets to reuse a permanent key)", flush=True)
    print(f'Test     : curl {url}/v1/models -H "Authorization: Bearer <API_KEY>"', flush=True)
    print("=" * 62 + "\n", flush=True)

# =============================================================================== main
def main() -> None:
    logger.info("Initializing Gemma 4 (12B) server...")

    gpus = detect_gpus()
    has_gpu = False
    
    if gpus:
        has_gpu = True
        logger.info("GPUs Detected: %s", ", ".join(f"{g['name']} ({g['mem_mb']} MiB)" for g in gpus))
    else:
        logger.warning("No NVIDIA GPU detected! System will fall back to CPU mode.")

    kill_stale_servers()
    binary = ensure_llama_server(gpus, has_gpu)
    model_path = download_model()

    backend = Backend(binary, model_path, gpus, has_gpu)
    tunnel = Tunnel(GATEWAY_PORT)

    def cleanup(*_a) -> None:
        try: tunnel.stop()
        finally: backend.stop()

    atexit.register(cleanup)
    try: signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except (ValueError, OSError): pass

    try:
        backend.start()
        threading.Thread(target=backend.supervise, daemon=True).start()

        config = uvicorn.Config(create_app(backend), host=GATEWAY_HOST, port=GATEWAY_PORT, log_level="warning",
                                loop="asyncio", http="h11", access_log=False, timeout_keep_alive=75)
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(60):
            if getattr(server, "started", False): break
            time.sleep(0.5)
        else: raise RuntimeError(f"Gateway failed to start on {GATEWAY_HOST}:{GATEWAY_PORT} (port in use?)")

        url = tunnel.start()
        if url: print_banner(url, has_gpu)
        else: logger.error("No public tunnel available. Reachable locally on %s:%s", GATEWAY_HOST, GATEWAY_PORT)
        if SELFTEST: threading.Thread(target=selftest, daemon=True).start()

        while not backend.gave_up:
            time.sleep(5)
            if tunnel.dead():
                logger.warning("Tunnel is down - restarting (the public URL may change)")
                new_url = tunnel.start()
                if new_url: print_banner(new_url, has_gpu)
    except KeyboardInterrupt: logger.info("Interrupted - shutting down")
    finally: cleanup()

if __name__ == "__main__":
    main()
