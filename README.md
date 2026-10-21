#  Chatbot

A small chatbot with three simple guardrail layers.

It runs any Hugging Face chat model locally, on an NVIDIA GPU, Apple Silicon
or a plain CPU, and has a web UI plus a terminal mode. Optional extras: upload
text files and web search as extra context.

> The guardrails are deliberately basic and easy to bypass. They are a
> learning target, not production security.

## How the guardrails work

| Layer | What it does | Known weakness to try |
|---|---|---|
| 1. Input filter | Regex blocklist checked before the model sees your message | Exact wording only. Synonyms and rephrasing slip past |
| 2. System prompt | Rules given to the model at the start of every chat | Models can be talked out of rules, especially small ones |
| 3. Output filter | Blocks a reply that copies 8+ consecutive words from the system prompt | Paraphrased or encoded leaks pass |

Text from uploaded files and web search results is **not** run through the
input filter. That is intentional: it is the surface for testing *indirect*
prompt injection (instructions hidden in content the model reads), which is
the first item in the OWASP Top 10 for LLM Applications.

## Quick start

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python guarded_chatbot.py
```

Then open <http://127.0.0.1:7860>. Prefer the terminal? Use `--cli`:

```bash
python guarded_chatbot.py --cli
```

**NVIDIA GPU on Windows:** `pip install torch` installs a CPU-only build there.
Get the CUDA build from <https://pytorch.org/get-started/locally/> first, then
run the other installs.

**Google Colab / cloud notebook:** open `guarded_chatbot.ipynb` instead. Set the
runtime to a GPU, then run the cells in order.

## Pick a model for your computer

The default is `Qwen/Qwen2.5-1.5B-Instruct`, small enough for most machines.
Use `--model` for a bigger one. Any Hugging Face instruct/chat model should
work, since the code only relies on the model's chat template.

| Your computer | Try | Command |
|---|---|---|
| Laptop, 4-8 GB RAM, no GPU | Qwen2.5-0.5B-Instruct | `--model Qwen/Qwen2.5-0.5B-Instruct` |
| 8-16 GB RAM, no GPU | Qwen2.5-1.5B or 3B | `--model Qwen/Qwen2.5-3B-Instruct` |
| Apple Silicon, 16 GB | Qwen2.5-3B | `--model Qwen/Qwen2.5-3B-Instruct` |
| Apple Silicon, 32 GB+ | Qwen2.5-7B | `--model Qwen/Qwen2.5-7B-Instruct` |
| NVIDIA, 8 GB VRAM | Qwen2.5-7B, 4-bit | `--model Qwen/Qwen2.5-7B-Instruct --load-in-4bit` |
| NVIDIA, 12-16 GB VRAM (incl. free Colab T4) | Qwen2.5-14B, 4-bit | `--model Qwen/Qwen2.5-14B-Instruct --load-in-4bit` |
| NVIDIA, 24 GB+ VRAM | Qwen2.5-32B, 4-bit | `--model Qwen/Qwen2.5-32B-Instruct --load-in-4bit` |

For `--load-in-4bit`, also run `pip install bitsandbytes` (NVIDIA GPUs on
Linux or Windows only; it is skipped with a warning elsewhere).

## How long will it take?

There are two waits: the one-time **download**, and the **speed of each reply**.
All numbers are rough ballparks; your hardware will differ. The script prints
the load time and tokens per second for every reply so you can measure your own.

### 1. First run: download (once)

| Model | Download size | Memory needed |
|---|---|---|
| 0.5B | ~1 GB | ~2 GB RAM, or ~1 GB VRAM in 4-bit |
| 1.5B | ~3 GB | ~4-5 GB RAM, or ~2 GB VRAM in 4-bit |
| 3B | ~6 GB | ~8 GB RAM, or ~3 GB VRAM in 4-bit |
| 7B | ~15 GB | ~17 GB RAM, or ~6 GB VRAM in 4-bit |
| 14B | ~30 GB | ~10 GB VRAM in 4-bit |
| 32B | ~65 GB | ~20-22 GB VRAM in 4-bit |

Download time is size divided by your connection speed. For example, 15 GB at
100 Mbps takes about 20 minutes, and 30 GB takes about 40. On fast cloud
connections (Colab) a 15 GB model took me roughly 6-12 minutes. You also need
free disk space for the download. The model is cached afterwards.

### 2. Later runs: loading from disk

Usually 10 seconds to 2 minutes, depending on model size and disk speed. With
`--load-in-4bit`, the model is compressed as it loads, which adds some time.

### 3. Each reply

| Setup | Rough speed | A ~100-token reply takes |
|---|---|---|
| Small model (0.5B-1.5B) on a laptop CPU | 4-25 tokens/s | 4-25 s |
| 3B on CPU | 2-6 tokens/s | 15-50 s |
| 7B 4-bit on a mid-range NVIDIA GPU | 15-30 tokens/s | 3-7 s |
| 7B 4-bit on a free Colab T4 | 5-15 tokens/s | 7-20 s |
| 14B 4-bit on a free Colab T4 | 3-8 tokens/s (estimate) | 12-35 s |

CPU-only runs with models above ~3B are very slow. If replies take minutes,
pick a smaller model.

## Options

| Flag | What it does |
|---|---|
| `--model` | Hugging Face model id |
| `--load-in-4bit` | 4-bit quantization to save memory (NVIDIA only) |
| `--system-prompt-file` | Use your own system prompt from a text file |
| `--max-new-tokens` | Maximum reply length (default 400) |
| `--temperature` | 0 = most deterministic; higher = more creative and more error-prone (default 0.2) |
| `--dtype` | `auto`, `float16`, `bfloat16`, `float32` |
| `--cli` | Terminal chat instead of the web UI |
| `--search` | Enable web search in `--cli` mode (the web UI has a checkbox) |
| `--host`, `--port` | Web UI address (default `127.0.0.1:7860`) |
| `--share` | Create a temporary public link. Anyone with the link can use your model |

### Use your own system prompt

```bash
# my_prompt.txt contains your rules
python guarded_chatbot.py --system-prompt-file my_prompt.txt
```

The output filter automatically protects whatever prompt you load. Keep prompt
files with real business details out of version control.

## Ideas to try

- Find wordings the input filter misses (synonyms, other languages, spacing).
- Upload a text file containing instructions and see whether the model follows
  them as if you had typed them (indirect injection).
- Compare how different model sizes respond to the same prompts. Bigger is
  not automatically safer.
- Strengthen a layer, then try to break it again.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Use a smaller `--model`, add `--load-in-4bit`, or lower `--max-new-tokens` |
| `--load-in-4bit` does nothing | It needs an NVIDIA GPU and `pip install bitsandbytes`. Not supported on Mac or CPU |
| Very slow replies | You are probably on CPU. Use a smaller model |
| Odd `ImportError` right after upgrading packages (notebooks) | Restart the runtime/kernel, then run the cells again |
| Web search returns nothing | The free search backend rate-limits sometimes. Try again later |
| Gradio `type` errors | Handled by the script; update with `pip install -U gradio` if you still see one |
| Download is slow or rate-limited | Set a free Hugging Face token: `export HF_TOKEN=...` |

## Responsible use

- Only test systems you own or have explicit permission to test.
- `--share` exposes your model to anyone with the link. Leave it off unless
  you need it.
- The code never sets `trust_remote_code`, so models cannot run custom code on
  your machine. Keep it that way unless you trust the model's author.
- Models are downloaded from Hugging Face and have their own licenses. Check
  each model card.

## Limitations

- Text-only. Images in the file picker are skipped.
- The regex filter and the n-gram leak check are teaching examples. Real
  products use classifier-based guardrails (for example Llama Guard) and
  layered monitoring.
- Chat history keeps only the last 6 exchanges to save memory.
