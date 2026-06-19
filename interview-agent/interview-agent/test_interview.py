"""
Scripted test harness for interview_agent.py.

Instead of talking to the agent with your voice, you give it a *script*: a list
of things the candidate "says" and pauses between them. The pauses let you test
the interviewer's patience — a pause longer than the agent's away-timeout will
trigger its "are you still there?" re-prompts and, if long enough, its graceful
close-out.

It runs the real agent logic with a real LLM, but over TEXT only (no microphone,
no speakers, no LiveKit room), using the framework's built-in `session.run()`
test entry point. At the end it prints and saves the same report the live agent
would produce.

Run it:
    python test_interview.py

You still need model credentials, exactly like the live agent (see CONFIG below).
"""

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
    inference,
)

# Reuse the actual program under test. Importing it does NOT start the agent;
# it just gives us the classes, the report writer, and the timing constants so
# the test stays in sync with the real behavior.
from interview_agent import (
    INTERVIEWER_NAME,
    MAX_REPROMPTS,
    USER_AWAY_TIMEOUT,
    InterviewData,
    IntroductionAgent,
    write_report,
)

load_dotenv()

# ============================================================================
# CONFIG  --  the model used to drive the test.
# Must match whatever interview_agent.py uses, since the LLM decides how the
# interviewer reacts. Option A (LiveKit inference gateway) is the default.
# ============================================================================
def make_test_llm():
    # Option A: LiveKit inference gateway (needs LIVEKIT_API_KEY etc.)
    return inference.LLM("openai/gpt-4.1-mini")

    # Option B: direct OpenAI plugin (needs OPENAI_API_KEY). If you switched
    # interview_agent.py to Option B, switch here too:
    # from livekit.plugins import openai
    # return openai.LLM(model="gpt-4o-mini")


# ============================================================================
# THE SCRIPT  --  this is what you edit to design a test.
#   Say("...")   the candidate says this; the agent gets a turn to respond.
#   Pause(n)     n seconds of silence. A pause longer than USER_AWAY_TIMEOUT
#                (currently {USER_AWAY_TIMEOUT}s) makes the agent check in;
#                staying silent through MAX_REPROMPTS ({MAX_REPROMPTS}) re-prompts
#                makes it end the interview.
# ============================================================================
@dataclass
class Say:
    text: str


@dataclass
class Pause:
    seconds: float


SCRIPT: list = [
    # --- Stage 1: introduction ---
    Say("Hi, I'm Alex. I've been a backend engineer for about five years."),
    Pause(3),  # short, polite pause — the agent should simply wait.
    Say("Mostly Python and Go, building payment and billing systems."),

    # --- Stage 2: experience ---
    Say("At my last job I led the migration of a monolith to microservices."),
    Pause(USER_AWAY_TIMEOUT + 5),  # long pause — should trigger ONE check-in.
    Say("Sorry about that, I'm back. I also mentored two junior engineers."),
    Say("A challenge I'm proud of: cutting our API latency by sixty percent."),

    # The agent should wrap up on its own once it has enough; if you'd rather
    # test the patience time-out instead, replace the lines above the wrap-up
    # with a single long Pause to let all re-prompts elapse, e.g.:
    #   Pause(USER_AWAY_TIMEOUT * (MAX_REPROMPTS + 2)),
]
# ============================================================================


# A simple holder so the inactivity handler and the script runner can share the
# "are we currently checking in on a silent candidate?" task.
class _State:
    inactivity_task: asyncio.Task | None = None
    closed: bool = False


def _print_turn(speaker: str, text: str) -> None:
    print(f"\n{speaker}: {text}", flush=True)


async def main() -> None:
    state = _State()

    # Text-only session: just an LLM, no STT/TTS/room. `session.run()` feeds
    # text turns directly. user_away_timeout is what powers the patience test.
    session = AgentSession[InterviewData](
        llm=make_test_llm(),
        userdata=InterviewData(),
        user_away_timeout=USER_AWAY_TIMEOUT,
    )

    # Print every committed turn as it happens — this is how we see the agent's
    # re-prompts that fire *during* a pause, not just its replies to our input.
    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if item.type == "message" and item.role == "assistant":
            _print_turn(INTERVIEWER_NAME, item.text_content.replace("\n", " "))
        elif item.type == "agent_handoff":
            print("\n   [stage change]", flush=True)

    @session.on("close")
    def _on_close(ev: CloseEvent) -> None:
        state.closed = True
        print(f"\n--- session closed ({ev.reason}) ---", flush=True)

    # --- Inactivity / patience handling -----------------------------------
    # This mirrors the wiring in interview_agent.py's entrypoint. Keep the two
    # in sync (or, if you refactor the program to expose a shared helper, call
    # that here instead).
    async def check_in_on_candidate() -> None:
        for _ in range(MAX_REPROMPTS):
            session.userdata.missed_prompts += 1
            await session.generate_reply(
                instructions=(
                    "The candidate has gone quiet. Warmly check whether they're still "
                    "there and offer to repeat the question. Keep it to one short sentence."
                )
            )
            await asyncio.sleep(USER_AWAY_TIMEOUT)
        await session.generate_reply(
            instructions=(
                "It seems we've lost the candidate. Politely note that you'll wrap up for "
                "now and they're welcome to reconnect. Say a brief goodbye."
            ),
            allow_interruptions=False,
        )
        session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "away":
            if state.inactivity_task is None or state.inactivity_task.done():
                print(f"\n   [candidate silent > {USER_AWAY_TIMEOUT}s — agent checking in]",
                      flush=True)
                state.inactivity_task = asyncio.create_task(check_in_on_candidate())

    # --- Run the script ----------------------------------------------------
    async with session:
        # capture_run=True so the opening greeting (on_enter) is driven before
        # we start feeding input.
        await session.start(IntroductionAgent(), capture_run=True)

        for step in SCRIPT:
            if state.closed:
                print("\n   [agent already ended the interview — stopping script]", flush=True)
                break

            if isinstance(step, Pause):
                print(f"\n   ...candidate pauses for {step.seconds:g}s...", flush=True)
                await asyncio.sleep(step.seconds)

            elif isinstance(step, Say):
                # The candidate is "back" — cancel any in-progress check-in,
                # just like real user activity would.
                if state.inactivity_task is not None and not state.inactivity_task.done():
                    state.inactivity_task.cancel()
                    state.inactivity_task = None

                _print_turn("CANDIDATE", step.text)
                await session.run(user_input=step.text)

        # Give any final agent speech a moment to commit before we report.
        await asyncio.sleep(0.5)

    # --- Report ------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f"Times candidate went silent: {session.userdata.missed_prompts}", flush=True)
    write_report(session)  # saves the Markdown report under ./interview_reports/
    print("Report written to ./interview_reports/", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
