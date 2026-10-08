<div align="center">

# 🤖 Gemma 4 (12B) — OpenAI-Compatible Server on Google Colab

**Run Gemma 4 12B (GGUF) on a free Colab T4 and expose it as an OpenAI-compatible API.**



**شغّل Gemma 4 12B على Colab T4 المجاني واعرضه كواجهة API متوافقة مع OpenAI.**



** لاستخدامه في VS  ينصح باضافة CLINE  وتشغيله باستخدام LLAMA **


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Backend](https://img.shields.io/badge/Backend-llama.cpp-green)
![API](https://img.shields.io/badge/API-OpenAI%20compatible-black)

[English](#english) · [العربية](#العربية)

</div>

## ⚡ Run in Colab in 2 lines · التشغيل في Colab بسطرين

Select a **T4 GPU** runtime (**Runtime → Change runtime type**), then paste this into a single Colab cell:

<div dir="rtl">

اختر بيئة تشغيل **T4 GPU** (**Runtime ← Change runtime type**) ثم الصق هذا في خلية واحدة في Colab:

</div>

```python
!wget -q -O gemma4_server.py https://raw.githubusercontent.com/abd19990044-commits/gemma412bcolab/main/gemma4_server.py
%run gemma4_server.py
```

When the banner appears, copy the **Base URL** and **API key** into your client — details in [Quick Start](#quick-start-google-colab).

<div dir="rtl">

عند ظهور اللوحة انسخ **Base URL** و **API key** إلى تطبيقك — التفاصيل في [البدء السريع](#البدء-السريع-google-colab).

</div>

---

# English

## Overview

A single, self-contained Python script that:

1. Downloads a prebuilt CUDA `llama-server` binary (from Google Drive via `gdown`) and the Gemma 4 12B GGUF model (from Hugging Face).
2. Starts `llama-server` in the background.
3. Puts a hardened **FastAPI gateway** in front of it (API key, rate limiting, tool-call parsing, SSE keep-alive).
4. Publishes the gateway through an **ngrok** or **Cloudflare** tunnel, so any OpenAI-compatible client can use it.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, and other `/v1/*` routes proxied to llama.cpp.
- **Automatic setup** — GPU detection, model download, binary download and extraction.
- **Automatic `LD_LIBRARY_PATH`** — every folder containing `.so` files in the extracted archive (plus common CUDA paths) is added before launching the binary.
- **API-key authentication** — `Authorization: Bearer <key>` or `x-api-key`; a random key is generated if you don't set one.
- **Tool / function calling** — a fallback parser understands `<tool_call>` blocks, `<function=...>` blocks and Markdown JSON blocks, and returns standard OpenAI `tool_calls`.
- **Streaming with keep-alive** — SSE comments are sent while waiting, so tunnels don't drop long requests.
- **Request queue** — `PARALLEL` slots with a queue timeout.
- **Per-IP rate limiting**, open CORS, and a maximum request body size.
- **Supervisor** — restarts `llama-server` if it crashes (gives up after 5 crashes in 30 minutes).
- **Self-test** on startup (plain chat + tool calling).
- **Multi-GPU** support (tensor split)****

## Architecture

```text
Client (OpenAI SDK / curl / any app)
        │  HTTPS
        ▼
ngrok  or  Cloudflare Tunnel
        │
        ▼
FastAPI Gateway :8001
  auth · rate limit · CORS · queue · tool-call parser · SSE keep-alive
        │  HTTP (localhost)
        ▼
llama-server :8000
  Gemma 4 12B (GGUF) on CUDA
```

## Requirements

- Linux x86_64 with Python 3.10+ (designed for Google Colab).
- An NVIDIA GPU (tested target: **Colab T4, 16 GB**). Without a GPU it fail.
- Internet access to Google Drive, Hugging Face and GitHub (for `cloudflared`).
- Python packages are installed automatically: `fastapi`, `uvicorn`, `httpx`, `huggingface_hub`, `gdown` (and `pyngrok` if an ngrok token is set).

## Quick Start (Google Colab)

1. Open a Colab notebook and select **Runtime → Change runtime type → T4 GPU**.
2. (Optional) Add secrets in **🔑 Secrets** and enable notebook access:

   | Secret | Purpose |
   |---|---|
   | `API_KEY` | Your permanent API key (otherwise a random one is generated each run) |
   | `HF_TOKEN` | Hugging Face token (only if the repo needs authentication / to avoid rate limits) |
   | `NGROK_AUTH_TOKEN` | Use ngrok instead of Cloudflare |

3. Download the script from GitHub and run it **inside the notebook** with `%run`:

   ```python
   !wget -q -O gemma4_server.py https://raw.githubusercontent.com/abd19990044-commits/gemma412bcolab/main/gemma4_server.py
   %run gemma4_server.py
   ```

   > Use `%run` rather than `!python`: `%run` executes in the notebook's own process, so **Colab Secrets** (`API_KEY`, `HF_TOKEN`, `NGROK_AUTH_TOKEN`) can be read. A script started with `!python` runs in a separate process that generally cannot read Colab Secrets (it would silently fall back to a random API key and Cloudflare). To use `!python`, forward the secrets first — see below.

4. Wait for the banner (the first run downloads the binary and the model, which takes a few minutes):

   ```text
   ==============================================================
   GEMMA 4 (12B) SYSTEM IS LIVE [GPU ACCELERATED]
   Base URL : https://xxxx.trycloudflare.com/v1
   Model    : gemma-4-12b-agent
   API key  : sk-...
   ==============================================================
   ```

5. Use the **Base URL**, **Model** and **API key** in your client.

To override settings, set environment variables in a cell **before** running the script:

```python
import os
os.environ["CTX_SIZE"] = "16384"
os.environ["ENABLE_THINKING"] = "0"
```

If you prefer `!python`, forward your Colab Secrets to environment variables first:

```python
import os
from google.colab import userdata
for k in ("API_KEY", "HF_TOKEN", "NGROK_AUTH_TOKEN"):
    try: os.environ[k] = userdata.get(k)
    except Exception: pass   # secret not defined

!python gemma4_server.py
```

### What happens on startup

1. Detects NVIDIA GPUs (`nvidia-smi`).
2. Kills any stale `llama-server` on the backend port.
3. Finds or downloads `llama-server` and fixes file permissions.
4. Downloads the GGUF model (skipped if it already exists).
5. Starts `llama-server` and waits until `/health` is OK.
6. Starts the gateway and opens the public tunnel.
7. Prints the banner and runs the self-test.

## Configuration

All settings are environment variables (booleans accept `1/true/yes/on`).

### Binary & model

| Variable | Default | Description |
|---|---|---|
| `GDRIVE_BIN_URL` | built-in link | Google Drive link to the archive containing `llama-server` (+ `.so` libraries) |
| `LLAMA_SERVER_BIN` | – | Path to an existing `llama-server`; skips the download |
| `LLAMA_CPP_DIR` | `/content/llama.cpp` | Where the archive is extracted |
| `MODEL_REPO` | `unsloth/gemma-4-12B-it-qat-GGUF` | Hugging Face repository |
| `GGUF_FILE` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | GGUF file name |
| `DOWNLOAD_DIR` | `/content/models` | Model directory |
| `FORCE_DOWNLOAD` | `0` | Re-download the model even if present |
| `SERVED_NAME` | `gemma-4-12b-agent` | Model id exposed by the API |

### Inference

| Variable | Default | Description |
|---|---|---|
| `CTX_SIZE` | `32768` | Context window (lower it if you run out of VRAM) |
| `PARALLEL` | `1` | Number of parallel slots (`-np`) |
| `KV_CACHE_TYPE` | `f16` | KV cache type (`-ctk/-ctv`); quantized types may require flash attention |
| `TENSOR_SPLIT` | `auto` | Multi-GPU split (`auto` = proportional to VRAM) |
| `LLAMA_EXTRA_ARGS` | `-fa off` | Extra arguments passed to `llama-server` |
| `LLAMA_VERBOSE` | `0` | Print every llama-server log line |
| `ENABLE_THINKING` | *(empty)* | `1` / `0` sends `chat_template_kwargs.enable_thinking`; empty = model default |
| `DEFAULT_MAX_TOKENS` | `4096` | Used when the client doesn't set `max_tokens` |

### Gateway

| Variable | Default | Description |
|---|---|---|
| `BACKEND_PORT` | `8000` | llama-server port (localhost only) |
| `GATEWAY_PORT` | `8001` | Public gateway port (behind the tunnel) |
| `GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `TOOL_MODE` | `buffered` | `buffered`: requests with `tools` are generated in full, parsed, then replayed as SSE (reliable tool calls). Any other value: real token streaming |
| `RATE_LIMIT_PER_MIN` | `120` | Requests per minute per IP (`0` disables) |
| `KEEPALIVE_SEC` | `15` | SSE keep-alive interval |
| `QUEUE_TIMEOUT` | `600` | Max seconds a request waits for a free slot |
| `UPSTREAM_READ_TIMEOUT` | `900` | Max seconds to wait for llama-server |
| `MAX_BODY_MB` | `32` | Maximum request body size |
| `SELFTEST` | `1` | Run the startup self-test |

### Secrets & tunnel

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | random `sk-...` | Key clients must send |
| `HF_TOKEN` | – | Hugging Face token |
| `NGROK_AUTH_TOKEN` | – | If set, ngrok is tried first, then Cloudflare as a fallback |
| `NGROK_DOMAIN` | – | Reserved ngrok domain |
| `CLOUDFLARED_BIN` | `/tmp/cloudflared` | Where `cloudflared` is stored |

> Secrets (`API_KEY`, `HF_TOKEN`, `NGROK_AUTH_TOKEN`) are read from environment variables first, then Colab Secrets.

## API Usage

### Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| `GET` | `/health` | No | Gateway and backend status |
| `GET` | `/v1/models` | Yes | Lists the served model |
| `POST` | `/v1/chat/completions` | Yes | Chat, streaming, tool calling |
| `GET/POST` | `/v1/*` | Yes | Any other route is proxied to llama-server |

### curl

```bash
curl https://<your-tunnel-url>/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gemma-4-12b-agent",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

### Python (OpenAI SDK) with tool calling

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-tunnel-url>/v1", api_key="<API_KEY>")

resp = client.chat.completions.create(
    model="gemma-4-12b-agent",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(resp.choices[0].message)
```

Streaming (`stream=True`) is supported. Works with any OpenAI-compatible client (Open WebUI, Continue, Cline, LangChain, etc.).

## Using with VS Code Extensions

Any VS Code extension that supports an **OpenAI-compatible** provider only needs the three values printed in the startup banner:

| Field | Value |
|---|---|
| Base URL / API Base | `https://xxxx.trycloudflare.com/v1` — copy it from the banner; it **must end with `/v1`** |
| API key | `sk-...` — the key from the banner (or your `API_KEY` secret) |
| Model ID | `gemma-4-12b-agent` — must match `SERVED_NAME` |

Before configuring an extension, make sure the server answers:

```bash
curl https://xxxx.trycloudflare.com/v1/models -H "Authorization: Bearer <API_KEY>"
```

### Continue

1. Install **Continue** from the Extensions view (`Ctrl+Shift+X`).
2. Open the Continue panel, click the model dropdown, then the ⚙️ gear next to **Local Config** to open `config.yaml`
   (or edit `~/.continue/config.yaml` directly; on Windows: `%USERPROFILE%\.continue\config.yaml`).
3. Add the model:

   ```yaml
   name: Local Config
   version: 1.0.0
   schema: v1
   models:
     - name: Gemma 4 12B (Colab)
       provider: openai
       model: gemma-4-12b-agent
       apiBase: https://xxxx.trycloudflare.com/v1
       apiKey: sk-xxxxxxxxxxxxxxxx
       roles:
         - chat
         - edit
         - apply
       capabilities:
         - tool_use          # lets Agent mode use tools
       defaultCompletionOptions:
         contextLength: 32768   # same as CTX_SIZE
         maxTokens: 4096
   ```

4. Save the file (Continue reloads automatically), select **Gemma 4 12B (Colab)** in the model dropdown and start chatting.

> `capabilities: [tool_use]` is declared explicitly because Continue's autodetection may not recognize a custom model name.

### Cline / Roo Code

1. Install **Cline** or **Roo Code** from the Extensions view.
2. Open the extension and click the ⚙️ **Settings** icon.
3. Set **API Provider** to **OpenAI Compatible** or **Llama**.
4. Fill in:
   - **Base URL**: `https://xxxx.trycloudflare.com/v1`
   - **API Key**: `sk-...`
   - **Model ID**: `gemma-4-12b-agent`
5. Under the model configuration, set the **context window** to your `CTX_SIZE`, the **max output tokens** to something like `4096`, and leave **image support** off (this script does not load a multimodal projector).
6. Save and start a task.

### Tips

- Keep `TOOL_MODE=buffered` (the default): agent modes then receive complete, well-formed `tool_calls`.
- Cloudflare quick-tunnel URLs **change on every restart** — update the Base URL in the extension each time, or use ngrok with `NGROK_DOMAIN` for a fixed URL.
- Set the `API_KEY` secret in Colab so the key doesn't change between runs.
- If you use an ngrok free domain and the extension receives an HTML page instead of JSON, add the header `ngrok-skip-browser-warning: true` in the extension's custom headers (in Continue: `requestOptions.headers`).
- A 12B model behind a tunnel is usually too slow for tab autocomplete; use it for chat, edit and agent tasks.
- Don't commit your extension config (it contains the API key) to a public repository.

## Troubleshooting

| Problem | Solution |
|---|---|
| `error while loading shared libraries` | Make sure the Drive archive contains the `.so` files next to the binary. Run with `LLAMA_VERBOSE=1` to see the full log |
| CUDA out of memory | Lower `CTX_SIZE` (e.g. `16384` or `8192`), keep `PARALLEL=1` |
| `gdown` fails / quota exceeded | Make sure the file is shared as *Anyone with the link*, retry later, or host the binary elsewhere and set `LLAMA_SERVER_BIN` |
| `503 Model is still loading` | Wait; the first start downloads and loads a multi-GB model |
| `401 Invalid API key` | Use the key printed in the banner, or set `API_KEY` in Colab Secrets |
| Tunnel URL changed | Cloudflare quick tunnels get a new URL on every restart; use ngrok with `NGROK_DOMAIN` for a stable URL |
| `llama-server` keeps crashing | Check the last log lines printed by the supervisor; the binary may not support this model's architecture |

## Security Notes

- ⚠️ **The prebuilt `llama-server` is downloaded from a Google Drive link and executed.** Only use binaries you built or trust. Prefer publishing it as a GitHub Release with a SHA-256 checksum, or build `llama.cpp` yourself and point `LLAMA_SERVER_BIN` to it.
- The tunnel makes the server **reachable from the whole internet**. Always keep API-key authentication enabled and never share the key or notebook output containing it.
- CORS is open (`*`) by default; restrict it in `create_app()` if you embed the API in a web app.
- Never commit tokens to the repository — use Colab Secrets or environment variables.

## License

The code in this repository is released under the **Apache License 2.0** — see [LICENSE](LICENSE).
The Gemma model is governed by its own license / terms of use; `llama.cpp` is MIT-licensed. Please review them before commercial use.

## Acknowledgements

[llama.cpp](https://github.com/ggml-org/llama.cpp) · [Unsloth](https://github.com/unslothai/unsloth) · [FastAPI](https://fastapi.tiangolo.com/) · [ngrok](https://ngrok.com/) · [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

---

<div dir="rtl">

# العربية

## نظرة عامة

سكربت بايثون واحد متكامل يقوم بما يلي:

1. تنزيل ملف `llama-server` الجاهز (مبني لـ CUDA) من Google Drive عبر `gdown`، وتنزيل نموذج Gemma 4 12B بصيغة GGUF من Hugging Face.
2. تشغيل `llama-server` في الخلفية.
3. وضع **بوابة FastAPI** أمامه توفّر مفتاح API، وتحديد معدل الطلبات، وتحليل استدعاءات الأدوات، ونبضات keep-alive للبث.
4. نشر البوابة عبر نفق **ngrok** أو **Cloudflare** ليتمكن أي عميل متوافق مع OpenAI من استخدامها.

## المميزات

- **واجهة متوافقة مع OpenAI** — `/v1/chat/completions` و `/v1/models` وبقية مسارات `/v1/*` تُمرَّر إلى llama.cpp.
- **إعداد تلقائي** — اكتشاف كرت الشاشة، وتنزيل النموذج، وتنزيل الملف التنفيذي وفك ضغطه.
- **ضبط `LD_LIBRARY_PATH` تلقائياً** — تُضاف كل المجلدات التي تحتوي ملفات `.so` في الأرشيف المفكوك (مع مسارات CUDA الشائعة) قبل تشغيل الملف التنفيذي.
- **مصادقة بمفتاح API** — عبر `Authorization: Bearer <key>` أو `x-api-key`، ويُولَّد مفتاح عشوائي إن لم تحدده.
- **دعم Tool / Function Calling** — محلل احتياطي يفهم كتل `<tool_call>` و `<function=...>` وكتل JSON بصيغة Markdown، ويعيدها بصيغة `tool_calls` القياسية.
- **بث (Streaming) مع keep-alive** — تُرسل نبضات SSE أثناء الانتظار حتى لا تقطع الأنفاق الطلبات الطويلة.
- **طابور طلبات** — عدد الخانات المتوازية `PARALLEL` مع مهلة انتظار.
- **تحديد معدل الطلبات لكل IP**، وCORS مفتوح، وحد أقصى لحجم الطلب.
- **مراقب للعملية** — يعيد تشغيل `llama-server` عند التعطل (ويتوقف بعد 5 أعطال خلال 30 دقيقة).
- **اختبار ذاتي** عند الإقلاع (محادثة عادية + استدعاء أداة).
- دعم **عدة كروت GPU (tensor split) **.

## البنية

<div dir="ltr">

```text
Client (OpenAI SDK / curl / any app)
        │  HTTPS
        ▼
ngrok  or  Cloudflare Tunnel
        │
        ▼
FastAPI Gateway :8001
  auth · rate limit · CORS · queue · tool-call parser · SSE keep-alive
        │  HTTP (localhost)
        ▼
llama-server :8000
  Gemma 4 12B (GGUF) on CUDA
```

</div>

## المتطلبات

- نظام Linux x86_64 مع Python 3.10 أو أحدث (مصمَّم للعمل على Google Colab).
- كرت NVIDIA (الهدف المُجرَّب: **Colab T4 بذاكرة 16 GB**).
- اتصال بالإنترنت يصل إلى Google Drive وHugging Face وGitHub (لتنزيل `cloudflared`).
- تُثبَّت الحزم تلقائياً: `fastapi` و `uvicorn` و `httpx` و `huggingface_hub` و `gdown` (و `pyngrok` إذا وُجد توكن ngrok).

## البدء السريع (Google Colab)

1. افتح دفتر Colab واختر **Runtime ← Change runtime type ← T4 GPU**.
2. (اختياري) أضف المفاتيح في **🔑 Secrets** وفعّل صلاحية الوصول من الدفتر:

   | المفتاح | الغرض |
   |---|---|
   | `API_KEY` | مفتاح API ثابت (وإلا يُولَّد مفتاح عشوائي في كل تشغيل) |
   | `HF_TOKEN` | توكن Hugging Face (عند الحاجة للمصادقة أو لتجنب حدود الطلبات) |
   | `NGROK_AUTH_TOKEN` | لاستخدام ngrok بدلاً من Cloudflare |

3. حمّل السكربت من GitHub وشغّله **داخل الدفتر** باستخدام `%run`:

<div dir="ltr">

```python
!wget -q -O gemma4_server.py https://raw.githubusercontent.com/abd19990044-commits/gemma412bcolab/main/gemma4_server.py
%run gemma4_server.py
```

</div>

> استخدم `%run` بدلاً من `!python`: فهو يعمل داخل عملية الدفتر نفسها، لذلك يمكنه قراءة **Colab Secrets** (`API_KEY` و `HF_TOKEN` و `NGROK_AUTH_TOKEN`). أما `!python` فيشغّل السكربت في عملية منفصلة لا تستطيع عادةً قراءة Colab Secrets (فيعود بصمت إلى مفتاح API عشوائي وإلى Cloudflare). وإن أردت استخدام `!python` فمرّر المفاتيح أولاً — انظر أدناه.

4. انتظر ظهور اللوحة التالية (في أول تشغيل يتم تنزيل الملف التنفيذي والنموذج، وقد يستغرق بضع دقائق):

<div dir="ltr">

```text
==============================================================
GEMMA 4 (12B) SYSTEM IS LIVE [GPU ACCELERATED]
Base URL : https://xxxx.trycloudflare.com/v1
Model    : gemma-4-12b-agent
API key  : sk-...
==============================================================
```

</div>

5. استخدم **Base URL** و **Model** و **API key** في تطبيقك.

لتغيير الإعدادات، عرّف متغيرات البيئة في خلية **قبل** تشغيل السكربت:

<div dir="ltr">

```python
import os
os.environ["CTX_SIZE"] = "16384"
os.environ["ENABLE_THINKING"] = "0"
```

</div>

وإذا فضّلت `!python` فمرّر مفاتيح Colab Secrets إلى متغيرات البيئة أولاً:

<div dir="ltr">

```python
import os
from google.colab import userdata
for k in ("API_KEY", "HF_TOKEN", "NGROK_AUTH_TOKEN"):
    try: os.environ[k] = userdata.get(k)
    except Exception: pass   # secret not defined

!python gemma4_server.py
```

</div>

### ماذا يحدث عند التشغيل؟

1. اكتشاف كروت NVIDIA عبر `nvidia-smi`.
2. إنهاء أي `llama-server` قديم يستخدم منفذ الخلفية.
3. البحث عن `llama-server` أو تنزيله وضبط صلاحياته.
4. تنزيل نموذج GGUF (يُتخطى إن كان موجوداً).
5. تشغيل `llama-server` وانتظار نجاح `/health`.
6. تشغيل البوابة وفتح النفق العام.
7. طباعة اللوحة وتشغيل الاختبار الذاتي.

## الإعدادات

كل الإعدادات عبارة عن متغيرات بيئة (القيم المنطقية تقبل `1/true/yes/on`).

### الملف التنفيذي والنموذج

| المتغير | الافتراضي | الوصف |
|---|---|---|
| `GDRIVE_BIN_URL` | رابط مدمج | رابط Google Drive للأرشيف الذي يحتوي `llama-server` (+ مكتبات `.so`) |
| `LLAMA_SERVER_BIN` | – | مسار `llama-server` موجود مسبقاً؛ يتخطى التنزيل |
| `LLAMA_CPP_DIR` | `/content/llama.cpp` | مكان فك الأرشيف |
| `MODEL_REPO` | `unsloth/gemma-4-12B-it-qat-GGUF` | مستودع Hugging Face |
| `GGUF_FILE` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | اسم ملف GGUF |
| `DOWNLOAD_DIR` | `/content/models` | مجلد النموذج |
| `FORCE_DOWNLOAD` | `0` | إعادة تنزيل النموذج حتى لو كان موجوداً |
| `SERVED_NAME` | `gemma-4-12b-agent` | اسم النموذج الظاهر في الـ API |

### الاستدلال (Inference)

| المتغير | الافتراضي | الوصف |
|---|---|---|
| `CTX_SIZE` | `32768` | حجم نافذة السياق (قلّله إذا نفدت ذاكرة VRAM) |
| `PARALLEL` | `1` | عدد الخانات المتوازية (`-np`) |
| `KV_CACHE_TYPE` | `f16` | نوع KV cache (`-ctk/-ctv`)؛ الأنواع المضغوطة قد تتطلب flash attention |
| `TENSOR_SPLIT` | `auto` | توزيع النموذج على عدة كروت (`auto` = بنسبة VRAM) |
| `LLAMA_EXTRA_ARGS` | `-fa off` | معاملات إضافية تُمرَّر إلى `llama-server` |
| `LLAMA_VERBOSE` | `0` | طباعة كل أسطر سجل llama-server |
| `ENABLE_THINKING` | *(فارغ)* | القيمة `1` أو `0` ترسل `chat_template_kwargs.enable_thinking`؛ الفارغ = سلوك النموذج الافتراضي |
| `DEFAULT_MAX_TOKENS` | `4096` | تُستخدم عندما لا يحدد العميل `max_tokens` |

### البوابة (Gateway)

| المتغير | الافتراضي | الوصف |
|---|---|---|
| `BACKEND_PORT` | `8000` | منفذ llama-server (محلي فقط) |
| `GATEWAY_PORT` | `8001` | منفذ البوابة (خلف النفق) |
| `GATEWAY_HOST` | `127.0.0.1` | عنوان ربط البوابة |
| `TOOL_MODE` | `buffered` | `buffered`: الطلبات التي تحتوي `tools` تُولَّد كاملة ثم تُحلَّل وتُعاد كبث SSE (استدعاءات أدوات موثوقة). أي قيمة أخرى: بث حقيقي للرموز |
| `RATE_LIMIT_PER_MIN` | `120` | عدد الطلبات في الدقيقة لكل IP (`0` لتعطيله) |
| `KEEPALIVE_SEC` | `15` | فاصل نبضات SSE |
| `QUEUE_TIMEOUT` | `600` | أقصى مدة (ثوانٍ) ينتظرها الطلب لخانة فارغة |
| `UPSTREAM_READ_TIMEOUT` | `900` | أقصى مدة انتظار لرد llama-server |
| `MAX_BODY_MB` | `32` | أقصى حجم لجسم الطلب |
| `SELFTEST` | `1` | تشغيل الاختبار الذاتي عند الإقلاع |

### المفاتيح والنفق

| المتغير | الافتراضي | الوصف |
|---|---|---|
| `API_KEY` | `sk-...` عشوائي | المفتاح الذي يجب أن يرسله العملاء |
| `HF_TOKEN` | – | توكن Hugging Face |
| `NGROK_AUTH_TOKEN` | – | إن وُجد تُجرَّب ngrok أولاً ثم Cloudflare كبديل |
| `NGROK_DOMAIN` | – | نطاق ngrok محجوز |
| `CLOUDFLARED_BIN` | `/tmp/cloudflared` | مكان حفظ `cloudflared` |

> تُقرأ المفاتيح (`API_KEY` و `HF_TOKEN` و `NGROK_AUTH_TOKEN`) من متغيرات البيئة أولاً، ثم Colab Secrets.

## استخدام الـ API

### المسارات (Endpoints)

| الطريقة | المسار | مصادقة | ملاحظات |
|---|---|---|---|
| `GET` | `/health` | لا | حالة البوابة والخلفية |
| `GET` | `/v1/models` | نعم | قائمة النموذج المتاح |
| `POST` | `/v1/chat/completions` | نعم | محادثة، بث، استدعاء أدوات |
| `GET/POST` | `/v1/*` | نعم | أي مسار آخر يُمرَّر إلى llama-server |

### curl

<div dir="ltr">

```bash
curl https://<your-tunnel-url>/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gemma-4-12b-agent",
        "messages": [{"role": "user", "content": "مرحباً!"}]
      }'
```

</div>

### Python (مكتبة OpenAI) مع استدعاء الأدوات

<div dir="ltr">

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-tunnel-url>/v1", api_key="<API_KEY>")

resp = client.chat.completions.create(
    model="gemma-4-12b-agent",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(resp.choices[0].message)
```

</div>

البث (`stream=True`) مدعوم. ويعمل مع أي عميل متوافق مع OpenAI (Open WebUI وContinue وCline وLangChain وغيرها).

## الاستخدام مع إضافات VS Code

أي إضافة في VS Code تدعم مزوّداً **متوافقاً مع OpenAI** تحتاج فقط إلى القيم الثلاث المطبوعة في لوحة الإقلاع:

| الحقل | القيمة |
|---|---|
| Base URL / API Base | `https://xxxx.trycloudflare.com/v1` — انسخه من اللوحة، ويجب أن **ينتهي بـ `/v1`** |
| API key | `sk-...` — المفتاح الظاهر في اللوحة (أو قيمة `API_KEY` التي حددتها) |
| Model ID | `gemma-4-12b-agent` — ويجب أن يطابق `SERVED_NAME` |

قبل إعداد الإضافة، تأكد أن الخادم يستجيب:

<div dir="ltr">

```bash
curl https://xxxx.trycloudflare.com/v1/models -H "Authorization: Bearer <API_KEY>"
```

</div>

### إضافة Continue

1. ثبّت **Continue** من واجهة الإضافات (`Ctrl+Shift+X`).
2. افتح لوحة Continue، واضغط على قائمة النماذج، ثم على أيقونة ⚙️ بجانب **Local Config** لفتح ملف `config.yaml`
   (أو عدّل الملف مباشرة: `~/.continue/config.yaml`، وفي ويندوز: `%USERPROFILE%\.continue\config.yaml`).
3. أضف النموذج:

<div dir="ltr">

```yaml
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: Gemma 4 12B (Colab)
    provider: openai
    model: gemma-4-12b-agent
    apiBase: https://xxxx.trycloudflare.com/v1
    apiKey: sk-xxxxxxxxxxxxxxxx
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use          # lets Agent mode use tools
    defaultCompletionOptions:
      contextLength: 32768   # same as CTX_SIZE
      maxTokens: 4096
```

</div>

4. احفظ الملف (تعيد Continue تحميل الإعدادات تلقائياً)، ثم اختر **Gemma 4 12B (Colab)** من قائمة النماذج وابدأ المحادثة.

> تم تحديد `capabilities: [tool_use]` صراحةً لأن الاكتشاف التلقائي في Continue قد لا يتعرف على اسم نموذج مخصص.

### إضافتا Cline / Roo Code

1. ثبّت **Cline** أو **Roo Code** من واجهة الإضافات.
2. افتح الإضافة واضغط على أيقونة ⚙️ **Settings**.
3. اختر **API Provider** ← **OpenAI Compatible**.
4. املأ الحقول:
   - **Base URL**: `https://xxxx.trycloudflare.com/v1`
   - **API Key**: `sk-...`
   - **Model ID**: `gemma-4-12b-agent`
5. في إعدادات النموذج اضبط **context window** على قيمة `CTX_SIZE`، و**max output tokens** على قيمة مثل `4096`، واترك دعم الصور **معطّلاً** (هذا السكربت لا يحمّل multimodal projector).
6. احفظ وابدأ مهمة جديدة.

### نصائح

- أبقِ `TOOL_MODE=buffered` (الافتراضي)؛ بذلك تستلم أوضاع الـ Agent استدعاءات أدوات `tool_calls` كاملة وسليمة.
- روابط Cloudflare السريعة **تتغير عند كل إعادة تشغيل** — حدّث Base URL في الإضافة كل مرة، أو استخدم ngrok مع `NGROK_DOMAIN` للحصول على رابط ثابت.
- عيّن `API_KEY` في Colab Secrets حتى لا يتغير المفتاح بين التشغيلات.
- إذا كنت تستخدم نطاق ngrok المجاني واستلمت الإضافة صفحة HTML بدلاً من JSON، فأضف الترويسة `ngrok-skip-browser-warning: true` في الترويسات المخصصة للإضافة (في Continue: `requestOptions.headers`).
- نموذج 12B خلف نفق غالباً بطيء جداً للإكمال التلقائي أثناء الكتابة (tab autocomplete)؛ استخدمه للمحادثة والتعديل ومهام الـ Agent.
- لا ترفع ملف إعدادات الإضافة (يحتوي مفتاح API) إلى مستودع عام.

## حل المشاكل

| المشكلة | الحل |
|---|---|
| `error while loading shared libraries` | تأكد أن أرشيف Drive يحتوي ملفات `.so` بجانب الملف التنفيذي، وشغّل مع `LLAMA_VERBOSE=1` لرؤية السجل الكامل |
| نفاد ذاكرة CUDA | قلّل `CTX_SIZE` (مثلاً `16384` أو `8192`) وأبقِ `PARALLEL=1` |
| فشل `gdown` / تجاوز الحصة | تأكد أن الملف مشارَك بصيغة *Anyone with the link*، أعد المحاولة لاحقاً، أو استضف الملف في مكان آخر وحدد `LLAMA_SERVER_BIN` |
| `503 Model is still loading` | انتظر قليلاً؛ أول تشغيل ينزّل ويحمّل نموذجاً بحجم عدة جيجابايت |
| `401 Invalid API key` | استخدم المفتاح المطبوع في اللوحة أو عيّن `API_KEY` في Colab Secrets |
| تغيّر رابط النفق | روابط Cloudflare السريعة تتغير عند كل إعادة تشغيل؛ استخدم ngrok مع `NGROK_DOMAIN` لرابط ثابت |
| تعطّل `llama-server` المتكرر | راجع آخر أسطر السجل التي يطبعها المراقب؛ قد لا يدعم الملف التنفيذي بنية هذا النموذج |

## ملاحظات أمنية

- ⚠️ **يتم تنزيل `llama-server` الجاهز من رابط Google Drive وتنفيذه.** استخدم فقط ملفات تنفيذية بنيتها بنفسك أو تثق بها. يُفضَّل نشره كـ GitHub Release مع بصمة SHA-256، أو بناء `llama.cpp` بنفسك وتمرير مساره عبر `LLAMA_SERVER_BIN`.
- النفق يجعل الخادم **متاحاً للإنترنت بأكمله**. أبقِ مصادقة مفتاح API مفعّلة دائماً، ولا تشارك المفتاح أو مخرجات الدفتر التي تحتويه.
- إعداد CORS مفتوح (`*`) افتراضياً؛ قيّده داخل `create_app()` إذا كنت ستستخدم الـ API من تطبيق ويب.
- لا ترفع التوكنات إلى المستودع أبداً — استخدم Colab Secrets أو متغيرات البيئة.

## الترخيص

شيفرة هذا المستودع مرخّصة بموجب **Apache License 2.0** — راجع ملف [LICENSE](LICENSE).
نموذج Gemma له رخصة وشروط استخدام خاصة به، و`llama.cpp` بترخيص MIT. يرجى مراجعتها قبل أي استخدام تجاري.

## شكر وتقدير

[llama.cpp](https://github.com/ggml-org/llama.cpp) · [Unsloth](https://github.com/unslothai/unsloth) · [FastAPI](https://fastapi.tiangolo.com/) · [ngrok](https://ngrok.com/) · [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

</div>
