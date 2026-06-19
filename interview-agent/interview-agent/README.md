# Voice Interview Agent

A voice-driven AI that conducts a realistic job interview for a **Software Engineer**
position. It greets the candidate, asks them to introduce themselves, explores their
past experience with natural follow-up questions, and then produces a written report
for a human to review afterward.

Built on the [LiveKit Agents](https://github.com/livekit/agents) framework (v1.6+).

---

## Features

- **Two-stage interview flow** — an introduction stage and a past-experience stage,
  implemented as separate agents that hand off to each other.
- **Natural, responsive interviewer** — professional but warm, and told to adapt its
  tone to the candidate (reassuring if they're nervous, matching their energy if
  they're enthusiastic).
- **Handles silence** — if the candidate goes quiet it checks in a few times, then
  ends the interview gracefully. Tuned for real interview think-time, and it stands
  down the moment the candidate speaks again.
- **Handles bad audio** — if speech can't be made out, or the candidate keeps getting
  cut off, it asks them to repeat / speak more clearly, and after repeated trouble it
  offers a typed fallback before ending.
- **Post-interview report** — writes a Markdown summary (candidate intro, experiences,
  engagement signals, a reviewer scorecard, and the full transcript) to
  `interview_reports/`.

## How it works

The interview is two `Agent` subclasses that hand off via a function-tool return,
following the framework's multi-agent pattern:

1. **`IntroductionAgent`** greets the candidate, introduces the interviewer, and asks
   them to introduce themselves. Once captured, it hands off.
2. **`ExperienceAgent`** explores past experience as plain conversation (no per-answer
   tool calls, so turns stay fast), then calls `conclude_interview` to wrap up.

Cross-cutting behavior lives in the `entrypoint` function at the session level:

- a **silence watchdog** (`user_away_timeout` + a check-in task),
- a **recognition/clarity handler** (reacts to empty transcripts and false
  interruptions, with a typed fallback), and
- a **report writer** that runs on shutdown.

Shared state travels in the `InterviewData` dataclass, and the experience list for the
report is extracted once at the end from the transcript.

## Project structure

```
.
├── interview_agent.py     # the agent (run this)
├── test_interview.py      # scripted, text-only test harness
├── requirements.txt
├── .env.example           # copy to .env and fill in
├── .gitignore
├── LICENSE
└── README.md
```

## Prerequisites

- Python 3.10+
- A microphone (for `console` mode)
- Model credentials — see the two options below.

## Setup

```bash
git clone https://github.com/Kel409/interview-agent.git
cd interview-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Download the local VAD / turn-detector model files (one time)
python interview_agent.py download-files

cp .env.example .env             # then edit .env with your keys
```

## Running it

The agent supports two ways to provide the speech/LLM models. Pick one and fill in the
matching keys in `.env`.

**Option A — LiveKit inference gateway (default).** The code ships using
`inference.STT/LLM/TTS`, which routes models through LiveKit's hosted gateway. Set
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` (from your LiveKit Cloud
project). No code changes needed.

**Option B — direct OpenAI plugin.** If you'd rather not use LiveKit Cloud, edit the
model lines in `interview_agent.py` to use the OpenAI plugin (`openai.STT()`,
`openai.LLM(model="gpt-4o-mini")`, `openai.TTS(voice="alloy")`) and set `OPENAI_API_KEY`
in `.env`.

Then run:

```bash
# Talk to it locally in your terminal (uses your mic + speakers)
python interview_agent.py console

# Or connect to a LiveKit room with hot reload (needs a frontend / the Agents Playground)
python interview_agent.py dev
```

`console` mode is the easiest way to try it. Live audio behaviors (interruptions,
silence handling) only really show up here, not in the text harness.

## Configuration

Everything adjustable is in the `CONFIG` block at the top of `interview_agent.py`:

| Setting | Default | What it does |
| --- | --- | --- |
| `POSITION` | `"Software Engineer"` | Role being interviewed for (woven into every prompt) |
| `COMPANY` | `"a leading global technology company"` | Company name used in prompts |
| `INTERVIEWER_NAME` | `"Jordan"` | The interviewer's name |
| `STT_MODEL` / `LLM_MODEL` / `TTS_MODEL` / `TTS_VOICE` | Deepgram / OpenAI / Cartesia | Which models/voice to use |
| `USER_AWAY_TIMEOUT` | `30.0` | Seconds of silence before the agent checks in (kept high for think-time) |
| `MAX_REPROMPTS` | `2` | How many times it checks in before ending |
| `MAX_RECOGNITION_FAILURES` | `3` | Failed/garbled attempts before offering a typed fallback / ending |
| `MIN_WORDS_FOR_VALID_ANSWER` | `1` | Transcripts shorter than this count as "didn't catch it" |
| `TROUBLE_DEBOUNCE` | `1.5` | Collapses bursts of trouble events into one |
| `FALSE_INTERRUPTION_TIMEOUT` | `3.5` | How long before a barge-in is judged a false interruption |
| `CLARITY_GRACE` | `4.0` | Wait before nudging "speak clearly," so a real answer can land first |

The interviewer's personality lives in the `PERSONA` string, also near the top.

## The report

When the interview ends (for any reason), a Markdown report is written to
`interview_reports/interview_<name>_<timestamp>.md` containing the candidate's
introduction, the experiences discussed, engagement signals (times silent, times
speech wasn't understood, etc.), a blank reviewer scorecard, and the full transcript.
The `interview_reports/` folder is git-ignored.

## Testing

`test_interview.py` drives the agent over **text only** (no mic/room) using a script
you edit at the top of the file:

```bash
python test_interview.py
```

Each step is a `Say("...")` (the candidate speaks) or a `Pause(n)` (n seconds of
silence). Pauses longer than `USER_AWAY_TIMEOUT` let you test the silence handling.
Note: text mode can't reproduce audio-specific behavior (interruptions, garbled
speech) — use `console` mode for those.

## Contributing

Contributions are welcome. A good starting point:

1. Fork the repo and create a branch.
2. Keep the code style consistent (the project targets readable, commented code; the
   underlying framework uses [ruff](https://docs.astral.sh/ruff/) with 100-char lines).
3. Test changes in `console` mode for anything audio-related.
4. Open a pull request describing the change and how you verified it.

Ideas for extension: add interview stages (e.g. a technical or behavioral round),
capture more structured fields in the report, add an AI-written assessment, or wire up
a web frontend.

## License

[MIT](Kenneth Lin).

## Acknowledgements

Built on [LiveKit Agents](https://github.com/livekit/agents) (Apache-2.0). See their
[docs](https://docs.livekit.io/agents/) for model options, deployment, and frontend
integration.
