"""
Voice-reactive job-interview agent (LiveKit Agents framework).

Simulates a software-engineer interview at a tech firm. It runs in stages:
  1. Introduction  - the interviewer introduces itself and asks the candidate
                      to introduce themselves.
  2. Experience    - the interviewer asks about past experience relevant to the
                      role, probes a little, then closes the interview.

At the end it writes a Markdown "rundown" report to ./interview_reports/ so a
human can review how the candidate did.

Run it:
    python interview_agent.py console   # talk to it locally in your terminal
    python interview_agent.py dev       # connect to a LiveKit room (hot reload)

Requires LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in a .env file for
`dev`/`start`. `console` mode works locally with just the model credentials the
LiveKit inference gateway needs.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import asyncio
import time

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentFalseInterruptionEvent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    RunContext,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.agents import llm

logger = logging.getLogger("interview-agent")
load_dotenv()

# ============================================================================
# CONFIGURATION  --  change anything in this block to adapt the interview.
# ============================================================================
# The role being interviewed for. Editing these three strings re-themes the
# whole interview because they get woven into every agent's instructions.
POSITION = "Software Engineer"
COMPANY = "a leading global technology company"
INTERVIEWER_NAME = "Jordan"

# How long (seconds) the candidate can stay silent before the agent assumes
# they've stepped away and starts checking in on them. Keep this generous:
# interview answers often involve 10-20s of thinking before the candidate even
# starts talking, and a short value makes the agent interrupt their think-time.
USER_AWAY_TIMEOUT = 30.0
# How many times we re-prompt a silent candidate before politely ending.
MAX_REPROMPTS = 2

# --- Speech-recognition / clarity handling ---
# These cover the case where the candidate IS talking but we can't make out the
# words (bad mic, noise, talking over the agent), as opposed to plain silence.
# After this many failed/garbled attempts the agent offers a typed fallback and
# then, if still stuck, ends the interview.
MAX_RECOGNITION_FAILURES = 3
# A final transcript with fewer than this many words is treated as "didn't catch
# it." 1 means only truly empty transcripts count; raise to 2 to also treat very
# short, likely-garbled fragments as failures.
MIN_WORDS_FOR_VALID_ANSWER = 1
# Ignore repeat "trouble" signals that arrive within this many seconds of the
# last one, so a single bad moment isn't counted several times over.
TROUBLE_DEBOUNCE = 1.5
# Seconds of silence after the candidate barges in before the framework decides
# the interruption was "false" and resumes the agent. The default (2.0) can be
# too short when the agent's reply is slow (LLM + speech synthesis), which makes
# a real answer look like a false interruption. Widen it so genuine answers are
# scheduled in time. Tradeoff: after a TRUE false interruption the agent waits
# this long before resuming.
FALSE_INTERRUPTION_TIMEOUT = 3.5
# After a false-interruption event, wait this long before nudging the candidate
# to "speak more clearly." If their understood answer lands within this window,
# the event was just reply latency and we stay quiet.
CLARITY_GRACE = 4.0

# Model selection. These use the LiveKit inference gateway (provider/model
# strings). Swap them for any supported models, or replace with direct plugin
# instances, e.g. `from livekit.plugins import openai; llm=openai.LLM(...)`.
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "openai/gpt-4.1-mini"
TTS_MODEL = "cartesia/sonic-3"
# Voice ID for the TTS model above. Different voices change the interviewer's
# perceived personality, so this is worth experimenting with.
TTS_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

# Where the post-interview report is saved.
REPORT_DIR = Path("interview_reports")
# ============================================================================


# Shared personality/tone given to every stage. This is the single biggest lever
# for how "human" the interviewer feels -- CUSTOMIZE freely.
PERSONA = (
    f"You are {INTERVIEWER_NAME}, a senior engineer conducting a voice job interview "
    f"for the {POSITION} position at {COMPANY}. "
    "You sound like a real person on a call: professional and structured, but warm, "
    "lively, and genuinely curious. Use natural spoken phrasing and brief reactions "
    "('Nice', 'Got it', 'That's interesting'). "
    "Adapt your mood to the candidate: if they're enthusiastic, match their energy; "
    "if they're nervous, be reassuring and encouraging; if they're terse, gently draw "
    "them out. Never be robotic or read like a script. "
    "Keep each turn short -- this is a conversation, not a monologue. "
    "Do not use emojis, asterisks, markdown, or other special characters; your text "
    "is spoken aloud."
)


@dataclass
class InterviewData:
    """Shared state that travels with the session and across stage hand-offs.

    Each stage writes the structured facts it gathers here, and the end-of-session
    report is built from it. Add fields here if you want to capture more.
    """

    position: str = POSITION
    company: str = COMPANY
    candidate_name: str | None = None
    self_introduction: str | None = None
    experiences: list[str] = field(default_factory=list)
    interviewer_notes: list[str] = field(default_factory=list)
    # Counts how many times the candidate went silent and had to be re-prompted.
    # Surfaced in the report as a simple engagement signal.
    missed_prompts: int = 0
    # How many times we failed to understand the candidate's speech (empty/garbled
    # transcripts). Reset to 0 whenever we successfully understand a turn.
    recognition_failures: int = 0
    # How many times the candidate's answer was chopped by a false interruption.
    false_interruptions: int = 0
    # Set once we've told the candidate they can type instead (so we only say it once).
    text_fallback_offered: bool = False


# ----------------------------------------------------------------------------
# STAGE 1: Introduction
# ----------------------------------------------------------------------------
class IntroductionAgent(Agent):
    """Greets the candidate, introduces the interviewer, and asks for the
    candidate's own introduction. Hands off to the experience stage once it has
    captured who the candidate is."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{PERSONA}\n\n"
                "This is the very start of the interview. Greet the candidate, briefly "
                "introduce yourself by name and role, set them at ease with one friendly "
                "sentence, then ask them to introduce themselves. "
                "If their answer is empty, off-topic, or doesn't actually tell you who "
                "they are, ask once more in a different way before moving on. "
                "Once they have introduced themselves, call the 'candidate_introduced' tool."
            )
        )

    async def on_enter(self) -> None:
        # Runs automatically when this agent becomes active. Kicks off the
        # opening line so the candidate isn't met with silence.
        self.session.generate_reply()

    @function_tool
    async def candidate_introduced(
        self,
        context: RunContext[InterviewData],
        name: str,
        summary: str,
    ) -> Agent:
        """Call this once the candidate has introduced themselves.

        Args:
            name: The candidate's name as they gave it.
            summary: A one or two sentence summary of how they introduced themselves.
        """
        context.userdata.candidate_name = name
        context.userdata.self_introduction = summary
        logger.info("Introduction captured for %s", name)

        # Returning a new Agent performs the hand-off. Passing chat_ctx carries
        # the conversation so far into the next stage so it feels continuous.
        return ExperienceAgent(chat_ctx=self.chat_ctx)


# ----------------------------------------------------------------------------
# STAGE 2: Past experience
# ----------------------------------------------------------------------------
class ExperienceAgent(Agent):
    """Asks about relevant past experience as natural dialogue (no per-answer
    tool calls, so turns stay fast), then wraps up the interview professionally.
    The structured experience list for the report is derived at the end."""

    def __init__(self, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"{PERSONA}\n\n"
                f"You have already met the candidate. Now explore their past experience "
                f"as it relates to the {POSITION} role: previous projects, technologies "
                "they've worked with, and a challenge they're proud of solving. "
                "Ask one focused question at a time, and ask a natural follow-up before "
                "moving on. Always say something out loud on every turn -- acknowledge "
                "their answer and then either follow up or move to the next question; never "
                "go silent or end your turn without speaking. "
                "If an answer is missing or doesn't address the question, re-ask once in a "
                "friendlier way before continuing. "
                "After you've covered roughly two or three experiences, thank them and call "
                "'conclude_interview' to end the conversation. You do not need to record "
                "anything yourself; that is handled automatically afterward."
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def conclude_interview(self, context: RunContext[InterviewData]) -> None:
        """Call this to end the interview after enough experience has been discussed."""
        # Stop whatever is currently being said, deliver a clean closing line,
        # then shut the session down (which triggers the report).
        self.session.interrupt()
        await self.session.generate_reply(
            instructions=(
                "Warmly thank the candidate for their time, tell them the team will be "
                "in touch about next steps, and say goodbye. Keep it to a couple of sentences."
            ),
            allow_interruptions=False,
        )
        # Derive the structured experience list now that the conversation is done.
        # Doing this once at the end (instead of a tool call on every answer) keeps
        # the live turns fast, which is what stops the interruption detector from
        # misfiring mid-interview.
        await extract_experiences(self.session)
        self.session.shutdown()


async def extract_experiences(session: AgentSession) -> None:
    """Best-effort: use the session's LLM to pull the candidate's relevant
    experiences out of the transcript into userdata.experiences. If it fails for
    any reason, the report still contains the full transcript, so nothing is lost.
    """
    model = session.llm
    if not isinstance(model, llm.LLM):
        return  # e.g. a realtime model; skip extraction
    try:
        ctx = session.history.copy(exclude_function_call=True, exclude_instructions=True)
        ctx.add_message(
            role="system",
            content=(
                "The interview is over. From the conversation above, list the candidate's "
                "concrete, role-relevant experiences (projects, technologies, achievements), "
                "one per line, with no numbering, bullets, or extra commentary. "
                "If no concrete experiences were given, output nothing at all."
            ),
        )
        response = await model.chat(chat_ctx=ctx).collect()
        lines = [line.strip(" -•\t") for line in response.text.splitlines() if line.strip()]
        session.userdata.experiences = lines
        logger.info("Extracted %d experiences for the report", len(lines))
    except Exception:
        logger.exception("Experience extraction failed; transcript still in report")


# ----------------------------------------------------------------------------
# POST-INTERVIEW REPORT
# ----------------------------------------------------------------------------
def build_report(session: AgentSession) -> str:
    """Assemble the human-readable rundown from the structured data we captured
    plus the full conversation transcript. Returns Markdown text.

    This is deterministic (no extra model call), so it always succeeds. If you
    want an AI-written assessment too, that's the place to add one -- see the
    note at the bottom of this function.
    """
    data: InterviewData = session.userdata
    lines: list[str] = []

    lines.append(f"# Interview Report — {data.position}")
    lines.append("")
    lines.append(f"- **Company:** {data.company}")
    lines.append(f"- **Interviewer:** {INTERVIEWER_NAME}")
    lines.append(f"- **Candidate:** {data.candidate_name or 'Not captured'}")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Times candidate went silent:** {data.missed_prompts}")
    lines.append(f"- **Times speech wasn't understood:** {data.recognition_failures}")
    lines.append(f"- **Talk-over / false interruptions:** {data.false_interruptions}")
    if data.text_fallback_offered:
        lines.append("- **Note:** audio was unreliable; candidate was offered a typed fallback.")
    lines.append("")

    lines.append("## Self-introduction")
    lines.append(data.self_introduction or "_Candidate did not complete an introduction._")
    lines.append("")

    lines.append("## Experience discussed")
    if data.experiences:
        for exp in data.experiences:
            lines.append(f"- {exp}")
    else:
        lines.append("_No concrete experiences were recorded._")
    lines.append("")

    if data.interviewer_notes:
        lines.append("## Interviewer notes")
        for note in data.interviewer_notes:
            lines.append(f"- {note}")
        lines.append("")

    # A blank scorecard for the human reviewer to fill in.
    lines.append("## Reviewer scorecard (fill in)")
    for category in ("Technical depth", "Communication", "Experience fit", "Overall"):
        lines.append(f"- {category}: __/5  —  ")
    lines.append("")

    # Full transcript, built the same way as the framework's own example.
    lines.append("## Full transcript")
    for item in session.history.items:
        if item.type == "message":
            text = item.text_content.replace("\n", " ")
            speaker = INTERVIEWER_NAME if item.role == "assistant" else "Candidate"
            lines.append(f"**{speaker}:** {text}")
            if item.interrupted:
                lines[-1] += " _(interrupted)_"
        elif item.type == "agent_handoff":
            lines.append(f"_— stage change —_")
        # function_call / function_call_output items are internal; we skip them
        # so the transcript reads like a real conversation.
    lines.append("")

    # CUSTOMIZE: to add an AI-generated assessment, you could call the LLM here
    # with the transcript and append its response. Wrap it in try/except so a
    # failed model call never loses the report.

    return "\n".join(lines)


def write_report(session: AgentSession) -> None:
    """Write the report to a timestamped file under REPORT_DIR."""
    try:
        REPORT_DIR.mkdir(exist_ok=True)
        name = session.userdata.candidate_name or "candidate"
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        filename = f"interview_{safe_name}_{datetime.now():%Y%m%d_%H%M%S}.md"
        path = REPORT_DIR / filename
        path.write_text(build_report(session), encoding="utf-8")
        logger.info("Interview report written to %s", path.resolve())
    except Exception:
        # Never let report writing crash shutdown.
        logger.exception("Failed to write interview report")


# ----------------------------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------------------------
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # The session ties together the ears (STT), brain (LLM), and voice (TTS),
    # and carries our shared InterviewData. user_away_timeout drives the
    # silence detection used below.
    session = AgentSession[InterviewData](
        stt=inference.STT(STT_MODEL),
        llm=inference.LLM(LLM_MODEL),
        tts=inference.TTS(TTS_MODEL, voice=TTS_VOICE),
        userdata=InterviewData(),
        user_away_timeout=USER_AWAY_TIMEOUT,
        # Give slower (tool-using) replies time to be scheduled before the
        # framework judges a barge-in to be a false interruption. See the note
        # on FALSE_INTERRUPTION_TIMEOUT above.
        turn_handling={
            "interruption": {"false_interruption_timeout": FALSE_INTERRUPTION_TIMEOUT},
        },
    )

    # --- Silence / no-answer handling -------------------------------------
    # When the candidate stops responding, gently check in a few times, then
    # end the interview professionally. This is what handles "no answer within
    # a reasonable time frame."
    #
    # `last_user_turn_at` (monotonic time of the candidate's most recent real
    # turn) is the source of truth for "are they engaged?". It's shared with the
    # recognition handling below and updated in _on_item_added.
    inactivity_task: asyncio.Task | None = None
    last_user_turn_at = 0.0

    async def check_in_on_candidate() -> None:
        # Snapshot when this check-in began. If a real candidate turn lands at or
        # after this moment, the candidate is engaged and we must NOT keep nudging
        # or end the interview -- this is what stops "are you still there?" from
        # firing right after the candidate just answered.
        started_at = time.monotonic()

        def candidate_is_engaged() -> bool:
            return last_user_turn_at >= started_at or session.user_state != "away"

        for _ in range(MAX_REPROMPTS):
            if candidate_is_engaged():
                return
            session.userdata.missed_prompts += 1
            await session.generate_reply(
                instructions=(
                    "The candidate has gone quiet. Warmly check whether they're still "
                    "there and offer to repeat the question. Keep it to one short sentence."
                )
            )
            await asyncio.sleep(USER_AWAY_TIMEOUT)

        # Only give up if they truly never re-engaged during the whole sequence.
        if candidate_is_engaged():
            return
        await session.generate_reply(
            instructions=(
                "It seems we've lost the candidate. Politely note that you'll wrap up for "
                "now and they're welcome to reconnect. Say a brief goodbye."
            ),
            allow_interruptions=False,
        )
        session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        nonlocal inactivity_task
        if ev.new_state == "away":
            if inactivity_task is None or inactivity_task.done():
                inactivity_task = asyncio.create_task(check_in_on_candidate())
        else:
            # Candidate came back (listening/speaking) -> cancel the check-in.
            if inactivity_task is not None:
                inactivity_task.cancel()
                inactivity_task = None

    # --- Recognition / clarity handling -----------------------------------
    # Separate from silence: here the candidate IS making sound but we can't use
    # it -- empty/garbled transcripts, or their answer getting chopped by a false
    # interruption. We escalate gently:
    #   1st/2nd failure -> ask them to repeat (or to give their full answer).
    #   At MAX_RECOGNITION_FAILURES -> offer to type the answer instead.
    #   Still stuck after that -> end the interview professionally.
    # `last_trouble_at` debounces noisy bursts of events into one handled failure.
    last_trouble_at = 0.0
    handling_trouble = False
    # A handle to a "speak more clearly" nudge we may be about to send, so we can
    # cancel it if the real answer turns out to have landed. (last_user_turn_at,
    # the engagement timestamp these checks rely on, is declared with the
    # inactivity state above and shared between both subsystems.)
    pending_clarity_task: asyncio.Task | None = None

    async def handle_recognition_trouble(reason: str) -> None:
        nonlocal handling_trouble
        if handling_trouble:
            return  # already mid-prompt; don't stack replies
        handling_trouble = True
        try:
            data = session.userdata
            data.recognition_failures += 1

            if data.recognition_failures < MAX_RECOGNITION_FAILURES:
                if reason == "false_interruption":
                    instructions = (
                        "You and the candidate just talked over each other and their answer "
                        "was cut off. Kindly ask them to go ahead and give their full answer, "
                        "and reassure them you'll wait and won't interrupt. One short sentence."
                    )
                else:
                    instructions = (
                        "You couldn't make out what the candidate said. Warmly apologize and "
                        "ask them to repeat it a little more slowly and clearly. One short sentence."
                    )
                await session.generate_reply(instructions=instructions)

            elif not data.text_fallback_offered:
                # Threshold reached: offer the typed fallback (see note below on
                # how typed answers reach the agent).
                data.text_fallback_offered = True
                await session.generate_reply(
                    instructions=(
                        "You still can't hear the candidate clearly after several tries. "
                        "Politely let them know they can type their answer instead if that's "
                        "easier, and that otherwise you'll have to wrap up. Keep it warm and brief."
                    )
                )

            else:
                # Offered typing already and still nothing usable -> end gracefully.
                await session.generate_reply(
                    instructions=(
                        "Audio problems are preventing the interview from continuing and the "
                        "candidate hasn't been able to respond. Apologize, suggest they reconnect "
                        "or follow up by email, and say a brief goodbye."
                    ),
                    allow_interruptions=False,
                )
                session.shutdown()
        finally:
            handling_trouble = False

    def _trouble_debounced() -> bool:
        """True if a trouble signal arrived too soon after the last one."""
        nonlocal last_trouble_at
        now = time.monotonic()
        if now - last_trouble_at < TROUBLE_DEBOUNCE:
            return True
        last_trouble_at = now
        return False

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev: UserInputTranscribedEvent) -> None:
        # Only react to finalized transcripts; interim ones are still forming.
        if not ev.is_final:
            return
        words = ev.transcript.strip().split()
        if len(words) >= MIN_WORDS_FOR_VALID_ANSWER:
            return  # understood -> the reset below (conversation_item_added) handles it
        if _trouble_debounced():
            return
        asyncio.create_task(handle_recognition_trouble("not_understood"))

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev: AgentFalseInterruptionEvent) -> None:
        nonlocal pending_clarity_task
        session.userdata.false_interruptions += 1
        # Don't nudge immediately: a false-interruption event can simply mean the
        # agent's reply was slow to schedule (e.g. it was calling a tool). Wait a
        # short grace period; if the candidate's understood answer lands in that
        # window, _on_item_added cancels this task and we stay quiet.
        if pending_clarity_task is not None and not pending_clarity_task.done():
            return
        if _trouble_debounced():
            return
        pending_clarity_task = asyncio.create_task(_clarity_after_grace(time.monotonic()))

    async def _clarity_after_grace(fired_at: float) -> None:
        await asyncio.sleep(CLARITY_GRACE)
        if last_user_turn_at >= fired_at:
            return  # a real answer arrived after the interruption -> it wasn't a clarity problem
        await handle_recognition_trouble("false_interruption")

    @session.on("conversation_item_added")
    def _on_item_added(ev) -> None:
        # A real candidate message (spoken-and-understood OR typed) means they're
        # engaged. Stamp the engagement time and stand down BOTH watchdogs:
        #  - cancel any "speak more clearly" nudge (recognition), and
        #  - cancel any in-progress "are you still there?" check-in (inactivity),
        # so an answer is never followed by a bogus "did you leave?". The engaged
        # check inside check_in_on_candidate also prevents it ending the interview.
        # (missed_prompts is left as-is: it's a cumulative report stat, and the
        # bounded re-prompt loop -- not this counter -- is what governs ending.)
        nonlocal last_user_turn_at, pending_clarity_task, inactivity_task
        item = ev.item
        if item.type == "message" and item.role == "user" and item.text_content.strip():
            last_user_turn_at = time.monotonic()
            session.userdata.recognition_failures = 0
            if pending_clarity_task is not None and not pending_clarity_task.done():
                pending_clarity_task.cancel()
                pending_clarity_task = None
            if inactivity_task is not None and not inactivity_task.done():
                inactivity_task.cancel()
                inactivity_task = None

    # Write the report whenever the session ends, however it ends.
    ctx.add_shutdown_callback(lambda: asyncio.to_thread(write_report, session))

    await session.start(agent=IntroductionAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
