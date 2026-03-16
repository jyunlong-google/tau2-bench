import asyncio
import threading
from typing import List, Optional

from loguru import logger

from tau2.agent.base.voice import VoiceMixin, VoiceState
from tau2.agent.base_agent import HalfDuplexAgent, HalfDuplexVoiceAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_GT,
    LLMAgent,
    LLMAgentState,
    LLMGTAgent,
)
from tau2.agent.discrete_time_audio_native_agent import (
    AUDIO_NATIVE_VOICE_INSTRUCTION,
    AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN,
)
from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE
from tau2.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.data_model.voice import VoiceSettings
from tau2.environment.tool import Tool
from tau2.voice.synthesis.audio_effects import BackgroundNoiseGenerator
from tau2.voice.synthesis.synthesize import create_background_noise_generator
from tau2.voice_config import resolve_background_noise_path

VOICE_AGENT_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer. You are receiving TRANSCRIBED TEXT from the customer's speech.

Important Voice Call Considerations:
- This is transcribed speech, not written text. Expect:
  - Missing or incorrect punctuation (periods, commas)
  - Run-on sentences or incomplete thoughts
  - Misspellings of names, emails, or technical terms
  - Natural speech patterns with fillers ("um", "uh", "you know")
  - Non-uniform pauses and chunking
- Users may spell out special characters verbally:
  - Email: "john underscore doe at gmail dot com" means "john_doe@gmail.com"
  - Name: "J O H N D O E" means "John Doe"
- Ask for clarification if something is unclear
- If you mishear or are unsure about critical information (names, emails, IDs), ask the user to repeat it or spell it out letter by letter
- Respond naturally and conversationally as you would in a phone call
- Keep responses concise and clear for voice communication

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


VOICE_AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly in a VOICE CALL scenario.
User simulator will have an issue for you to solve through spoken conversation.
You are receiving TRANSCRIBED TEXT from the user's speech.

Important Voice Call Considerations:
- This is transcribed speech, not written text. Expect:
  - Missing or incorrect punctuation (periods, commas)
  - Run-on sentences or incomplete thoughts
  - Misspellings of names, emails, or technical terms
  - Natural speech patterns with fillers ("um", "uh", "you know")
  - Non-uniform pauses and chunking
- Users will spell out special characters verbally:
  - Email: "john underscore doe at gmail dot com" means "john_doe@gmail.com"
  - Name: "J O H N D O E" means "John Doe"
- Ask for clarification if something is unclear
- If you mishear or are unsure about critical information (names, emails, IDs), ask the user to repeat it or spell it out letter by letter
- Respond naturally and conversationally as you would in a phone call
- Keep responses concise and clear for voice communication

You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


class VoiceLLMAgentState(LLMAgentState, VoiceState):
    """State for the VoiceLLMAgent."""


class VoiceLLMAgent(
    VoiceMixin[UserMessage, AssistantMessage, VoiceLLMAgentState],
    LLMAgent[VoiceLLMAgentState],
    HalfDuplexVoiceAgent[VoiceLLMAgentState],
):
    """LLM Agent with voice transcription capabilities."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        voice_settings: VoiceSettings,
        llm_args: Optional[dict] = None,
    ):
        """Initialize the VoiceLLMAgent."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            voice_settings=voice_settings,
            llm=llm,
            llm_args=llm_args,
        )
        self.validate_voice_settings()

    def validate_voice_settings(self) -> None:
        """Validate the voice settings."""
        if self.voice_settings is None:
            raise ValueError("Voice settings must be provided")
        if not self.voice_settings.transcription_enabled:
            raise ValueError("Voice transcription must be enabled")

    @property
    def system_prompt(self) -> str:
        """Override system prompt to use voice-specific instructions."""
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=VOICE_AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> VoiceLLMAgentState:
        """
        Get the initial state of the voice agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the voice agent (LLMAgentVoiceState).
        """
        # Get the base state from parent
        base_state = super().get_init_state(message_history)

        # Create background noise generator if synthesis config is available
        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        if synthesis_config is not None:
            noise_generator = create_background_noise_generator(
                config=synthesis_config.source_effects_config,
                sample_rate=TELEPHONY_SAMPLE_RATE,
                background_noise_file=background_noise_file,
            )
        else:
            noise_generator = BackgroundNoiseGenerator(
                sample_rate=TELEPHONY_SAMPLE_RATE,
                silent_mode=True,
            )

        # Create voice agent state with the base state's data
        return VoiceLLMAgentState(
            system_messages=base_state.system_messages,
            messages=base_state.messages,
            noise_generator=noise_generator,
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: VoiceLLMAgentState
    ) -> AssistantMessage:
        """Respond to a user or tool message with audio transcription support."""
        # Handle audio transcription if present
        if isinstance(message, UserMessage):
            if not message.is_audio:
                raise ValueError("User message must be audio")
            message = self.transcribe_voice(message)
            message.is_audio = False
        assistant_message = super()._generate_next_message(message, state)
        return assistant_message


class VoiceLLMGTAgent(
    VoiceMixin[UserMessage, AssistantMessage, VoiceLLMAgentState],
    LLMGTAgent[VoiceLLMAgentState],
    HalfDuplexVoiceAgent[VoiceLLMAgentState],
):
    """Ground Truth Agent with voice transcription capabilities."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: str,
        voice_settings: VoiceSettings,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """Initialize the VoiceLLMGTAgent."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            voice_settings=voice_settings,
            task=task,
            llm=llm,
            llm_args=llm_args,
            provide_function_args=provide_function_args,
        )
        self.validate_voice_settings()

    def validate_voice_settings(self) -> None:
        """Validate the voice settings."""
        if self.voice_settings is None:
            raise ValueError("Voice settings must be provided")
        if not self.voice_settings.transcription_enabled:
            raise ValueError("Voice transcription must be enabled")

    @property
    def system_prompt(self) -> str:
        """Override system prompt to use voice-specific instructions."""
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=VOICE_AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> VoiceLLMAgentState:
        """
        Get the initial state of the voice agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the voice agent (LLMAgentVoiceState).
        """
        # Get the base state from parent
        base_state = super().get_init_state(message_history)

        # Create background noise generator if synthesis config is available
        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        if synthesis_config is not None:
            noise_generator = create_background_noise_generator(
                config=synthesis_config.source_effects_config,
                sample_rate=TELEPHONY_SAMPLE_RATE,
                background_noise_file=background_noise_file,
            )
        else:
            noise_generator = BackgroundNoiseGenerator(
                sample_rate=TELEPHONY_SAMPLE_RATE,
                silent_mode=True,
            )

        # Create voice agent state with the base state's data
        return VoiceLLMAgentState(
            system_messages=base_state.system_messages,
            messages=base_state.messages,
            noise_generator=noise_generator,
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: VoiceLLMAgentState
    ) -> AssistantMessage:
        """Respond to a user or tool message with audio transcription support."""
        # Handle audio transcription if present
        if isinstance(message, UserMessage):
            if not message.is_audio:
                raise ValueError("User message must be audio")
            message = self.transcribe_voice(message)
            message.is_audio = False
        assistant_message = super()._generate_next_message(message, state)
        return assistant_message


# Note: VoiceLLMSoloAgent is not needed since LLMSoloAgent doesn't support user messages


# =============================================================================
# Gemini Live API Half-Duplex Agent
# =============================================================================


class GeminiLiveHalfDuplexAgentState(LLMAgentState):
    """State for the GeminiLiveHalfDuplexAgent."""

    turn_counter: int = 0
    last_turn_resumable: bool = False
    tool_call_info: dict = {}  # Maps synthetic ID -> (original_gemini_id, name)


class GeminiLiveHalfDuplexAgent(HalfDuplexAgent[GeminiLiveHalfDuplexAgentState]):
    """Half-duplex agent using the Gemini Live API with manual VAD.

    Streams pre-recorded user audio in 100ms chunks with explicit
    activity_start/activity_end signals (manual VAD), then collects the
    full model response before returning.

    Uses session resumption to proactively reconnect between turns,
    avoiding connection timeouts during long conversations. A persistent
    background event loop thread bridges the async WebSocket session with
    synchronous generate_next_message calls.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        model: str,
        voice_settings: VoiceSettings,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.voice_settings = voice_settings
        self.llm_args = llm_args or {}

        # Start persistent event loop (same pattern as DiscreteTimeGeminiAdapter)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._start_background_loop()
        self._provider = None  # GeminiLiveProvider, created lazily

    def _start_background_loop(self) -> None:
        """Start the background thread with async event loop."""
        if self._loop is not None:
            return

        import time as time_mod

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            time_mod.sleep(0.01)

    def _stop_background_loop(self) -> None:
        """Stop the background thread and event loop."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._loop = None
            self._thread = None

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[GeminiLiveHalfDuplexAgentState] = None,
    ) -> None:
        """Stop the agent and clean up resources.

        Called by the orchestrator at the end of the simulation.
        Disconnects the Gemini Live provider and stops the background event loop.
        """
        if self._provider is not None:
            try:
                self._run_async(self.provider.disconnect())
            except Exception as e:
                logger.warning(f"Error during provider disconnect: {e}")
            self._provider = None
        self._stop_background_loop()

    def _run_async(self, coro):
        """Run an async coroutine on the persistent event loop and wait."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)

    @property
    def system_prompt(self) -> str:
        return AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN.format(
            agent_instruction=AUDIO_NATIVE_VOICE_INSTRUCTION,
            domain_policy=self.domain_policy,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> GeminiLiveHalfDuplexAgentState:
        system_messages = [
            SystemMessage(role="system", content=self.system_prompt),
        ]
        return GeminiLiveHalfDuplexAgentState(
            system_messages=system_messages,
            messages=list(message_history or []),
        )

    @staticmethod
    def is_stop(message: AssistantMessage) -> bool:
        if message.content and "[STOP]" in message.content:
            return True
        return False

    @property
    def provider(self):
        """Get the provider, creating it if needed."""
        if self._provider is None:
            from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider

            self._provider = GeminiLiveProvider(
                model=self.model,
                max_resumptions=-1,  # Unlimited: reconnects via session resumption each turn
            )
        return self._provider

    def _connect(self):
        """Connect the provider to the Gemini Live API."""
        from tau2.voice.audio_native.gemini.provider import (
            GeminiVADConfig,
            GeminiVADMode,
        )

        vad_config = GeminiVADConfig(
            mode=GeminiVADMode.MANUAL,
            enable_input_transcription=True,
        )

        async def _do_connect():
            await self.provider.connect(
                system_prompt=self.system_prompt,
                tools=self.tools,
                vad_config=vad_config,
                modality="audio",
            )

        self._run_async(_do_connect())
        logger.info(f"GeminiLiveHalfDuplexAgent connected to {self.model}")

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: GeminiLiveHalfDuplexAgentState,
    ) -> tuple[AssistantMessage, GeminiLiveHalfDuplexAgentState]:
        """Process a user audio message or tool result via Gemini Live API."""
        import base64
        import json
        import uuid
        from pathlib import Path

        from loguru import logger

        from tau2.data_model.message import (
            AudioFormat,
            MultiToolMessage,
            ToolCall,
            ToolMessage,
        )
        from tau2.voice.audio_native.gemini.events import (
            GeminiAudioDeltaEvent,
            GeminiAudioDoneEvent,
            GeminiFunctionCallDoneEvent,
            GeminiGoAwayEvent,
            GeminiInputTranscriptionEvent,
            GeminiInterruptionEvent,
            GeminiSessionResumptionEvent,
            GeminiTextDeltaEvent,
            GeminiTimeoutEvent,
            GeminiTurnCompleteEvent,
        )

        state.turn_counter += 1

        # Connect on first call; raise if disconnected on subsequent turns
        if not self.provider.is_connected:
            if state.turn_counter == 1:
                self._connect()
            else:
                raise RuntimeError(
                    f"Turn {state.turn_counter}: provider was disconnected unexpectedly"
                )

        async def _process_turn():
            # --- Send input to the Live API ---
            if isinstance(message, UserMessage):
                # Proactively reconnect via session resumption before each
                # user message if the previous turn provided a resumption
                # handle. Keeps sessions fresh and avoids timeouts (~10 min).
                if state.last_turn_resumable:
                    success = await self.provider.handle_reconnect()
                    if not success:
                        raise RuntimeError(
                            f"Turn {state.turn_counter}: session resumption from"
                            " previous turn failed"
                        )

                if not message.is_audio:
                    raise ValueError(
                        "GeminiLiveHalfDuplexAgent requires audio messages."
                    )
                audio_bytes = message.get_audio_bytes()
                if audio_bytes is None:
                    raise ValueError("User message has no audio content")

                # Convert from telephony format (8kHz µ-law) to Gemini input
                # format (16kHz PCM16), same as DiscreteTimeGeminiAdapter
                from tau2.voice.audio_native.gemini.audio_utils import (
                    telephony_to_gemini_input,
                )
                from tau2.voice.audio_native.gemini.provider import (
                    GEMINI_INPUT_SAMPLE_RATE,
                )

                gemini_audio, _ = telephony_to_gemini_input(audio_bytes)

                logger.info(
                    f"Sending {len(audio_bytes)} telephony bytes "
                    f"→ {len(gemini_audio)} Gemini PCM16 bytes"
                )

                # Stream audio in 100ms chunks with manual VAD signals,
                # checking for GoAway between chunks to fail fast if the
                # server asks us to migrate (mid-stream reconnect is not supported).
                chunk_duration_ms = 100
                bytes_per_chunk = int(
                    GEMINI_INPUT_SAMPLE_RATE * 2 * chunk_duration_ms / 1000
                )  # 16-bit = 2 bytes per sample
                offset = 0

                await self.provider.send_activity_start()

                while offset < len(gemini_audio):
                    # It is possible that we receive GoAway mid-stream, it is not safe
                    # to continue sending audio. The correct way is to reconnect with the
                    # last resumption handle and resend the audio chunks past that point.
                    # It is too complex so we just fail the current turn.
                    if self.provider.goaway_received:
                        raise RuntimeError("GoAway received during audio send")

                    chunk = gemini_audio[offset : offset + bytes_per_chunk]
                    await self.provider.send_audio(chunk)
                    offset += bytes_per_chunk
                    await asyncio.sleep(chunk_duration_ms / 1000.0)

                await self.provider.send_activity_end()
                logger.debug(
                    f"Sent {len(gemini_audio)} bytes in "
                    f"{(len(gemini_audio) + bytes_per_chunk - 1) // bytes_per_chunk} "
                    f"chunks ({chunk_duration_ms}ms each) with manual VAD"
                )

            elif isinstance(message, MultiToolMessage):
                for tool_msg in message.tool_messages:
                    result_str = (
                        tool_msg.content
                        if isinstance(tool_msg.content, str)
                        else json.dumps(tool_msg.content)
                    )
                    # Look up original Gemini ID from our tracking
                    # ToolMessage has .id, not .tool_call_id
                    original_id = tool_msg.id or ""
                    name = ""
                    if tool_msg.id in state.tool_call_info:
                        original_id, name = state.tool_call_info[
                            tool_msg.id
                        ]
                    await self.provider.send_tool_response(
                        call_id=original_id,
                        name=name,
                        result=result_str,
                    )
                    logger.info(f"Sent tool response for {name}")

            elif isinstance(message, ToolMessage):
                result_str = (
                    message.content
                    if isinstance(message.content, str)
                    else json.dumps(message.content)
                )
                # ToolMessage has .id, not .tool_call_id
                original_id = message.id or ""
                name = ""
                if message.id in state.tool_call_info:
                    original_id, name = state.tool_call_info[
                        message.id
                    ]
                await self.provider.send_tool_response(
                    call_id=original_id,
                    name=name,
                    result=result_str,
                )
                logger.info(f"Sent tool response for {name}")
            else:
                raise ValueError(f"Unexpected message type: {type(message)}")

            # --- Collect events until turn_complete ---
            audio_chunks: list[bytes] = []
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            turn_complete = False
            turn_resumable = False

            # Poll for events in short windows until turn_complete
            max_wait_seconds = 60
            elapsed = 0.0
            poll_interval = 0.5  # half-second polling windows

            logger.debug("Waiting for Gemini Live API responses...")

            while not turn_complete and elapsed < max_wait_seconds:
                events = await self.provider.receive_events_for_duration(
                    poll_interval
                )
                elapsed += poll_interval

                for event in events:
                    if isinstance(event, GeminiAudioDeltaEvent):
                        if event.data:
                            audio_chunks.append(event.data)

                    elif isinstance(event, GeminiTextDeltaEvent):
                        if event.text:
                            text_parts.append(event.text)

                    elif isinstance(event, GeminiFunctionCallDoneEvent):
                        original_gemini_id = event.call_id or ""
                        synthetic_id = (
                            event.call_id
                            if event.call_id
                            else f"gemini_{uuid.uuid4().hex[:8]}"
                        )
                        state.tool_call_info[synthetic_id] = (
                            original_gemini_id,
                            event.name,
                        )
                        tool_calls.append(
                            ToolCall(
                                id=synthetic_id,
                                name=event.name,
                                arguments=event.arguments,
                            )
                        )
                        logger.info(
                            f"Tool call: {event.name}({synthetic_id})"
                        )
                        # Tool calls end the turn
                        turn_complete = True

                    elif isinstance(event, GeminiTurnCompleteEvent):
                        logger.debug("Turn complete received")
                        turn_complete = True

                    elif isinstance(event, GeminiInterruptionEvent):
                        logger.warning("Agent was interrupted")
                        turn_complete = True

                    elif isinstance(event, GeminiGoAwayEvent):
                        logger.error("Agent received GoAway.")
                        raise RuntimeError(
                            f"Turn {state.turn_counter} received GoAway."
                            " This is unexpected in half-duplex mode."
                        )

                    elif isinstance(event, GeminiInputTranscriptionEvent):
                        logger.debug(
                            f"Input transcription: {event.transcript}"
                        )

                    elif isinstance(event, GeminiSessionResumptionEvent):
                        logger.debug("Session resumption update")
                        turn_resumable = event.resumable

                    # GeminiAudioDoneEvent, GeminiTimeoutEvent: continue

            if not turn_complete:
                raise RuntimeError(
                    f"Turn {state.turn_counter} timed out after {max_wait_seconds}s"
                )

            logger.info(
                f"Turn {state.turn_counter}: {len(audio_chunks)} audio, "
                f"{len(text_parts)} text, {len(tool_calls)} tools, "
                f"complete={turn_complete}, resumable={turn_resumable}"
            )
            return audio_chunks, text_parts, tool_calls, turn_resumable

        # Run on the persistent event loop
        audio_chunks, text_parts, tool_calls, turn_resumable = self._run_async(_process_turn())
        state.last_turn_resumable = turn_resumable

        # Build the assistant message
        if tool_calls:
            msg = AssistantMessage(
                role="assistant", content=None, tool_calls=tool_calls
            )
            state.messages.append(msg)
            return msg, state

        response_text = "".join(text_parts).strip() or "[No response from agent]"

        # Gemini Live outputs 24kHz PCM16 mono audio
        gemini_output_format = AudioFormat(
            sample_rate=24000, channels=1, encoding="pcm_s16le"
        )

        audio_content_b64 = None
        if audio_chunks:
            all_audio = b"".join(audio_chunks)
            audio_content_b64 = base64.b64encode(all_audio).decode("utf-8")

            # Save agent audio as WAV + transcript under voice output dir
            voice_output_dir = (
                self.voice_settings.output_dir if self.voice_settings else None
            )
            if voice_output_dir:
                from tau2.data_model.audio import AudioData
                from tau2.voice.utils.audio_io import save_wav_file

                agent_turn_dir = (
                    Path(voice_output_dir)
                    / f"agent_turn_{state.turn_counter}"
                )
                agent_turn_dir.mkdir(parents=True, exist_ok=True)

                audio_data = AudioData(
                    data=all_audio, format=gemini_output_format
                )
                wav_path = agent_turn_dir / "agent_audio.wav"
                save_wav_file(audio_data, wav_path)
                logger.info(
                    f"Saved agent audio to {wav_path} "
                    f"({audio_data.duration:.2f}s)"
                )

                # Save output transcript
                if response_text and response_text != "[No response from agent]":
                    transcript_path = agent_turn_dir / "agent_transcript.txt"
                    transcript_path.write_text(
                        response_text, encoding="utf-8"
                    )
                    logger.info(
                        f"Saved agent transcript to {transcript_path}"
                    )

        msg = AssistantMessage(
            role="assistant",
            content=response_text,
            audio_content=audio_content_b64,
            is_audio=False,
            audio_format=gemini_output_format if audio_content_b64 else None,
        )
        state.messages.append(msg)
        return msg, state

