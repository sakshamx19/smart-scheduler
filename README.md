# Smart Scheduler

Voice-oriented meeting scheduler that chats with the user, parses natural time requests, and books meetings on Google Calendar.

Smart Scheduler is a LangGraph agent with a FastAPI server and a voice chat portal. It was designed and tested in KIRA, an agent studio, and exported from there on October 18, 2026.

## What it does

- **Works with a Google Calendar** (`find_free_slots`, `list_events`, `create_event`, `update_event`, `delete_event`): finds open times that fit how long, when and how the user wants to meet, offers the nearest alternatives when nothing fits, books meetings, and moves or cancels them.
- **Built-in tools:** `current_time`.
- **Talks.** Speak to it in the browser and it answers aloud, starting while the reply is still being written.
- **Asks before it acts.** Anything that changes something waits for the user's OK.

## Run it

Needs Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in the keys (see Configuration)
uvicorn server.main:app --reload --port 8000
```

Open http://localhost:8000, press the microphone and talk, or chat in the terminal with `python -m agent.cli`.

## Configuration

Every setting is an environment variable; `.env.example` lists them with comments.

| Variable | Needed | What it is |
|---|---|---|
| `LLM_PROVIDER` | yes | openai (any OpenAI-compatible endpoint) or anthropic. Default here: `openai`. |
| `OPENAI_API_KEY` | yes | Your key for the endpoint below. |
| `OPENAI_BASE_URL` | no | Empty for OpenAI itself. Azure: https://&lt;resource&gt;.openai.azure.com/openai/v1/. Default here: `https://ai-pai.services.ai.azure.com/openai/v1/`. |
| `OPENAI_MODEL` | yes | The model, or on Azure the deployment name. Default here: `gpt-5.1`. |
| `OPENAI_REASONING_EFFORT` | no | Reasoning models only: none or low answer fastest, which matters for voice. |
| `LLM_TEMPERATURE` | no | Optional. Some reasoning models accept only their default; leave it empty for those. |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | yes | The JSON key of saksham-kumar@tryingbard-388910.iam.gserviceaccount.com, on one line (or use the file below). |
| `GOOGLE_CALENDAR_ID` | yes | Shared with the service account (Make changes to events). Default here: `saksham.kumar82@gmail.com`. |
| `CALENDAR_TIMEZONE` | no | IANA name; the calendar's own time zone. Default here: `Asia/Kolkata`. |
| `WORK_START` | no | Start of working hours, used unless the user asks for other times. Default here: `09:00`. |
| `WORK_END` | no | End of working hours. Default here: `18:00`. |
| `WORK_DAYS` | no | Working days, separated by commas. Default here: `monday,tuesday,wednesday,thursday,friday`. |
| `CHECKPOINT_DB` | no | Conversations are kept here (SQLite); empty keeps them in memory. Default here: `data/agent.db`. |
| `ACCESS_CODE` | no | Set it to keep strangers off a public deployment; the portal asks for it once. |
| `STT_PROVIDER` | no | device (Whisper in the browser, as in KIRA: private, about 77 MB once) \| browser (Chrome and Edge, fastest) \| openai \| deepgram. Default here: `device`. |
| `TTS_PROVIDER` | no | device (Supertonic in the browser, as in KIRA: natural voice, about 262 MB once) \| browser (instant) \| openai \| elevenlabs. Default here: `device`. |
| `VOICE_LANGUAGE` | no | BCP 47 language for speech, e.g. en-US, en-IN, hi-IN. Default here: `en-US`. |
| `BROWSER_VOICE` | no | Optional: part of a browser voice's name to prefer, e.g. Aria or Google. |

## Google Calendar setup

The agent reaches the calendar as a Google **service account**, so it needs no sign-in screen and keeps working on
a server.

1. In [Google Cloud Console](https://console.cloud.google.com/), pick or create a project and enable the
   **Google Calendar API**.
2. Under **IAM & Admin → Service accounts**, use `saksham-kumar@tryingbard-388910.iam.gserviceaccount.com` (the account KIRA used) or create one, then
   **Keys → Add key → JSON**. Keep the file private.
3. In [Google Calendar](https://calendar.google.com/), open the settings of the calendar `saksham.kumar82@gmail.com` →
   **Share with specific people** → add the service account's email with **Make changes to events**.
4. Put the key in `GOOGLE_SERVICE_ACCOUNT_JSON` (the whole file on one line), or save it next to the project and
   set `GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json`. `.gitignore` already keeps such files out of git.

A service account cannot send invitations from a personal Gmail calendar. When the user names guests, the agent
still books the meeting and lists them in its description, and says so. (Google Workspace calendars can allow
invitations with domain-wide delegation.)

## How it works

### The graph

```
START ─► agent ─► asked for tools? ─ no ─► END
           ▲            │ yes
           │            ▼
         tools ◄──── review   (pauses for your approval when a call changes something)
```

`agent/graph.py` is a LangGraph `StateGraph` with three nodes. **agent** calls the model with the system prompt, the whole conversation and the tools. **review** runs when the model asks for tools: if a call changes something, it pauses the run with `interrupt()` and the server shows the user a one-line summary ("Book “Sync” on Tuesday, Oct 20, 2:00 PM – 3:00 PM"). **tools** runs the calls side by side; a declined call or a failing tool answers the model with what happened, so it can recover instead of crashing.

### Memory

Each conversation is a LangGraph thread kept by a checkpointer in SQLite (`CHECKPOINT_DB`), so it survives a restart. The model sees every earlier turn, which is how it keeps the meeting length from three messages ago when the user only changes the day. A paused approval lives in the same checkpoint: nothing waits in memory between the question and the answer.

### The prompt

`agent/prompts.py` holds the system prompt written in KIRA. On every turn `system_prompt()` appends what the tools know: the calendar's time zone, today, this week and next week, the month's last weekday, and the next four weeks of dates. Models are unreliable at date arithmetic; giving them the facts, and having the tools take local ISO times, removes a whole class of mistakes. When the user is speaking, it also asks for short spoken sentences.

### Finding times

`find_free_slots` does the scheduling in code, not in the model: it reads busy times from the calendar (ignoring events marked free or declined), keeps to working hours or the hours asked for, skips excluded days, keeps a buffer around meetings when asked, and spreads its suggestions across the window instead of offering 9:00, 9:30 and 10:00. When nothing fits it answers the nearest alternatives (other hours the same day, then later and earlier days), each with the reason, so the agent can say "Tuesday afternoon is full. Would Wednesday at 2 PM work?". `create_event` checks the slot is still free before booking. The logic is in `agent/tools/scheduling.py` and `google_calendar.py`, which have no LangChain dependency and are covered by `tests/test_scheduling.py`.

### Voice and latency

Speech runs on the user's own device, as in KIRA, where this agent was built: Whisper (base) turns speech into text and Supertonic turns text into speech, through transformers.js in Web Workers (`web/whisper.worker.js`, `web/tts.worker.js`). No audio or text leaves the browser for speech and nothing is paid per use. The models download once and the browser caches them, so the first conversation waits while they load (the portal shows the download). The server sends cross-origin isolation headers, which let WebAssembly use several threads; with them Supertonic makes a 4-second sentence in about 1.5 s on a laptop. A turn ends after one second of silence.

Whatever the providers, the portal streams the reply token by token and speaks it a sentence at a time, so the first words play while the model is still writing the rest. When the model calls a tool before saying anything, the portal says "One moment." at once. After each spoken turn it shows the measured delay from the end of the user's speech to the agent's first word.

| Provider | Speech to text (`STT_PROVIDER`) | Text to speech (`TTS_PROVIDER`) |
|---|---|---|
| `device` | Whisper base in the browser: private, free, about 77 MB once | Supertonic in the browser: natural voice, about 262 MB once |
| `browser` | The browser's own recognition (Chrome, Edge): nothing to download | The browser's own voices: instant |
| `openai` | `gpt-4o-mini-transcribe` on the server | `gpt-4o-mini-tts`, streamed |
| `deepgram` / `elevenlabs` | Deepgram Nova on the server | ElevenLabs Flash, streamed |

Every provider ships in the project, so switching is a change to `.env`, not a new export.

Approvals work by voice too: the agent reads the summary and asks "Shall I go ahead?". "Yes" approves, "no" declines, and anything else ("yes, but at 3") declines and passes the words on to the model.

To keep latency low: use a fast model, set `OPENAI_REASONING_EFFORT=none` (or `low`) for reasoning models, and deploy near the model's region with at least one instance kept warm.

## Deploy

The `Dockerfile` builds one container with the agent, its API and the portal.

**Google Cloud Run**

```bash
# .env as a YAML file for Cloud Run (values such as WORK_DAYS and the JSON key contain commas,
# which --set-env-vars would split). env.yaml holds your keys: it is git- and docker-ignored.
python -c "import json; from dotenv import dotenv_values; print('\n'.join(f'{k}: {json.dumps(v)}' for k, v in dotenv_values('.env').items() if v))" > env.yaml

gcloud run deploy smart-scheduler --source . --region us-central1 --allow-unauthenticated \
  --min-instances 1 --max-instances 1 --timeout 300 --env-vars-file env.yaml
```

Keys are better kept in Secret Manager (`--set-secrets OPENAI_API_KEY=openai-key:latest`). Set `ACCESS_CODE` on a
public URL, so only people you give the code to can use the agent and its calendar.

**Render, Railway or Fly.io**: create a web service from this folder (they find the `Dockerfile`), add the
variables from `.env.example`, and keep one instance always on.

Why one instance: conversations and paused approvals live in that instance's checkpointer. To run several, switch
to LangGraph's Postgres checkpointer (`langgraph-checkpoint-postgres`) in `agent/runtime.py`.

## Tests

```bash
python -m pytest -q
```

They need no keys or network: `tests/test_graph.py` drives the graph with a scripted model (tools, approval, decline, a failing tool, memory), and `tests/test_scheduling.py` runs the calendar tools against a fake calendar.

## Project layout

```
smart-scheduler/
  .dockerignore
  .env.example
  .gitignore
  Dockerfile
  README.md
  pytest.ini
  requirements.txt
  agent/
    __init__.py
    cli.py
    config.py
    graph.py
    llm.py
    prompts.py
    runtime.py
    tools/
      __init__.py
      builtin.py
      calendar.py
      google_auth.py
      google_calendar.py
      scheduling.py
  server/
    __init__.py
    main.py
    voice.py
  tests/
    test_graph.py
    test_scheduling.py
  web/
    app.js
    index.html
    styles.css
    tts.worker.js
    voice.js
    whisper.worker.js
```

## Design choices

- **An explicit `StateGraph` instead of a prebuilt agent**, so the approval step is a real node that can pause and resume, and the control flow can be read in one file and tested without a model (`tests/test_graph.py`).
- **Scheduling logic in plain Python, not in the prompt**: the model turns "sometime next week, not too early, not Wednesday" into parameters, and code finds the slots. That is deterministic, testable and fast, and leaves the model to do what it is good at: understanding the request and talking.
- **Approval as `interrupt()` with a checkpointer**: the paused run costs nothing while it waits and survives a restart, and the user always sees exactly the arguments that will run.
- **Errors become tool results**: a calendar outage or a bad argument is something the model can explain or retry, not a crashed conversation.
- **The model is chosen by environment variables** (`LLM_PROVIDER` and friends), so moving between OpenAI, Azure OpenAI, OpenRouter or Claude needs no code change.
- **A cascaded voice pipeline (speech to text → LangGraph → text to speech)** rather than a speech-to-speech model, so the same graph, tools and approvals serve voice and text, and each provider can be swapped.

## Limitations

- One conversation runs at a time per thread, and the state lives in one instance (see Deploy).
- On-device speech (`device`) downloads its models on first use (about 340 MB for both) and is slower on phones and in Safari (no multi-threading there); `browser` speech recognition needs Chrome or Edge. Pick per audience in `.env`.
- One calendar per agent, reached as a service account (see Google Calendar setup).
