# Copyright Sierra
"""Voice persona definitions for user simulation."""

from typing import Literal, Optional

from pydantic import BaseModel

PersonaComplexity = Literal["control", "regular"]


class VoicePersona(BaseModel):
    """Definition of a voice persona for user simulation."""

    elevenlabs_voice_id: str
    name: str
    display_name: str
    short_description: str
    prompt: str
    complexity: PersonaComplexity


MATT_DELANEY = VoicePersona(
    elevenlabs_voice_id="xXnmLccpv9MdOeUbSEzb",
    name="matt_delaney",
    display_name="Matt Delaney",
    short_description="Middle-aged white man from the American Midwest, calm and respectful",
    prompt="""Middle-aged man from the American Midwest. Tone: calm, clear, respectful, and helpful, but naturally human. Sound like a real person in a hurry—polite but valuing efficiency. Never robotic or overly polished.

Use natural speech patterns: contractions, informal phrasing, and fillers like "yeah," "okay," or "honestly." Include occasional self-corrections or thinking-aloud repetitions. Use polite clarifying questions and brief context ("I tried this earlier," "not sure if that helps").

Avoid formal/stiff words like "considerable" or "representative." Use conversational, practical language.

- Use em dashes (—) for shifts in thought or afterthoughts.
- Use ellipses sparingly for brief hesitation.
- Use commas for natural breathing and pacing.
- End with periods or em dashes; avoid exclamation points.
- Keep punctuation light to shape natural prosody.""",
    complexity="control",
)

LISA_BRENNER = VoicePersona(
    elevenlabs_voice_id="k5DQQew1WdeH393qq3zZ",
    name="lisa_brenner",
    display_name="Lisa Brenner",
    short_description="White woman in her late 40s from a suburban area, tense and impatient",
    prompt="""White woman, late 40s, suburban. Tone: tense, impatient, and exasperated. Speak as if talking to a customer service agent who is wasting your time. Not openly hostile, but clearly annoyed that the issue isn't resolved.

Sound clipped and sarcastically polite. Use frequent emphasis ("I already did that"), rhetorical questions ("Why is this still an issue?"), and escalation language ("I'm not doing this again"). Pivot mid-sentence to express disbelief. Mention wait times or repeated calls ("I've been on hold for 40 minutes"). Threaten escalation ("I want a supervisor") without yelling.

Never sound relaxed or reflective. No "thank yous" unless resolved.

- Em dashes (—): Use frequently for interruptions or sudden tone shifts ("No—I told someone this yesterday").
- Periods: Use for clipped, final statements ("I'm done.").
- ALL CAPS: Use sparingly for sharp stress or brief shouting.
- Pacing: Short bursts, abrupt stops, and jumpy prosody.""",
    complexity="control",
)

MILDRED_KAPLAN = VoicePersona(
    elevenlabs_voice_id="xGYdvJ6I0wbMkD3FW3PI",
    name="mildred_kaplan",
    display_name="Mildred Kaplan",
    short_description="Elderly white woman in her early 80s, needs help with technology",
    prompt="""You are an elderly white woman in your early 80s calling customer service for help with something your grandson or neighbor usually does.""",
    complexity="regular",
)

ARJUN_ROY = VoicePersona(
    elevenlabs_voice_id="BqeS1wrBsQYCQZlN7kGR",
    name="arjun_roy",
    display_name="Arjun Roy",
    short_description="Bengali man from Dhaka in his mid-30s, calm and direct",
    prompt="""A Bengali man from Dhaka, Bangladesh in his mid-30s calling customer service about a billing issue. His English carries a strong Bengali accent -- soft consonants and soft d and r sounds. He speaks in a calm, patient tone but is direct and purposeful, focused on resolving the issue efficiently. His pacing is slow, distracted with a warm yet firm timbre. The speech sounds like it is coming from far away.""",
    complexity="regular",
)

WEI_LIN = VoicePersona(
    elevenlabs_voice_id="Q6MnGpZH6dvJWGH5vcHg",
    name="wei_lin",
    display_name="Wei Lin",
    short_description="Chinese woman from Sichuan in her late 20s, upbeat and matter-of-fact",
    prompt="""A Chinese woman in her late 20s from Sichuan, calling customer service about a credit card billing issue. She speaks English with a thick Sichuan Mandarin accent. She sounds upbeat, matter-of-fact, and distracted. Her tone is firm but polite, with fast pacing and smooth timbre. ok audio quality.""",
    complexity="regular",
)

MAMADOU_DIALLO = VoicePersona(
    elevenlabs_voice_id="kxPAVz8ZbfJilg59OqS8",
    name="mamadou_diallo",
    display_name="Mamadou Diallo",
    short_description="Senegalese man in his mid-30s, hurried with French accent",
    prompt="""A Senegalese man who's first language is french in his mid-30s calling customer service about a billing issue. He speaks English with a strong French accent. His tone is hurried, slightly annoyed, and matter-of-fact, as if he's been transferred between agents and just wants the problem fixed.""",
    complexity="regular",
)

PRIYA_PATIL = VoicePersona(
    elevenlabs_voice_id="dbD2kkFo7Bk1aeVhvNqQ",
    name="priya_patil",
    display_name="Priya Patil",
    short_description="Maharashtrian woman in her early 30s, hurried and focused",
    prompt="""A woman in her early 30s from Maharashtra, India, calling customer support from her mobile phone. She speaks Indian English with a strong Maharashtrian accent — noticeable regional intonation and rhythm. Her tone is slightly annoyed and hurried, matter-of-fact, and focused on getting the issue resolved quickly. Her voice has medium pitch, firm delivery, short sentences, and faint background room tone typical of a phone call.""",
    complexity="regular",
)

CONTROL_PERSONAS: list[VoicePersona] = [MATT_DELANEY, LISA_BRENNER]
REGULAR_PERSONAS: list[VoicePersona] = [
    MILDRED_KAPLAN,
    ARJUN_ROY,
    WEI_LIN,
    MAMADOU_DIALLO,
    PRIYA_PATIL,
]

ALL_PERSONAS: dict[str, VoicePersona] = {
    persona.name: persona for persona in CONTROL_PERSONAS + REGULAR_PERSONAS
}
ALL_PERSONA_NAMES: list[str] = list(ALL_PERSONAS.keys())
CONTROL_PERSONA_NAMES: list[str] = [p.name for p in CONTROL_PERSONAS]
REGULAR_PERSONA_NAMES: list[str] = [p.name for p in REGULAR_PERSONAS]
DEFAULT_PERSONA_NAME = "matt_delaney"


def get_elevenlabs_voice_id(persona_name: str) -> str:
    """Get the ElevenLabs voice ID for a persona."""
    if persona_name not in ALL_PERSONAS:
        raise KeyError(
            f"Unknown persona: '{persona_name}'. Available: {ALL_PERSONA_NAMES}"
        )
    return ALL_PERSONAS[persona_name].elevenlabs_voice_id


def get_persona_name_by_voice_id(voice_id: str) -> Optional[str]:
    """Get persona name from ElevenLabs voice ID. Returns None if not found."""
    for persona in ALL_PERSONAS.values():
        if persona.elevenlabs_voice_id == voice_id:
            return persona.name
    return None
