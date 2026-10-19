"""Speech for the portal. Each direction has a provider, chosen in .env:

    STT_PROVIDER   device | browser | openai | deepgram       speech → text
    TTS_PROVIDER   device | browser | openai | elevenlabs     text → speech

"device" runs open models in the visitor's browser, as KIRA does: Whisper hears (web/whisper.worker.js) and
Supertonic speaks (web/tts.worker.js). Nothing leaves the device and nothing costs per use; the models download once.
"browser" uses the browser's own speech (the Web Speech API): nothing to download and the lowest latency, in Chrome
and Edge. The other providers run here, so their keys never reach the browser.

Text to speech is streamed: the portal asks for one sentence at a time while the reply is still being written,
and the browser starts playing the audio as its first bytes arrive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.config import env, require

router = APIRouter(tags=["voice"])
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_EXTENSIONS = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav"}


def stt_provider() -> str:
    return env("STT_PROVIDER", "device").lower()


def tts_provider() -> str:
    return env("TTS_PROVIDER", "device").lower()


def public_config() -> dict[str, Any]:
    return {"stt": stt_provider(), "tts": tts_provider(), "language": env("VOICE_LANGUAGE", "en-US"), "browser_voice": env("BROWSER_VOICE")}


def _openai_audio() -> tuple[str, str]:
    base = (env("OPENAI_AUDIO_BASE_URL") or env("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    return base, env("OPENAI_AUDIO_API_KEY") or require("OPENAI_API_KEY")


def _failed(provider: str, resp: httpx.Response, body: bytes) -> HTTPException:
    return HTTPException(502, f"{provider} answered {resp.status_code}: {body[:300].decode('utf-8', 'replace')}")


@router.post("/stt")
async def speech_to_text(request: Request) -> dict[str, str]:
    """The request body is the recording (audio/webm from MediaRecorder, usually)."""
    audio = await request.body()
    if not audio:
        raise HTTPException(422, "Send the recording as the request body.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "That recording is too long.")
    content_type = request.headers.get("content-type", "audio/webm").split(";")[0].strip()
    language = env("VOICE_LANGUAGE", "en-US")
    provider = stt_provider()
    async with httpx.AsyncClient(timeout=30) as client:
        if provider == "openai":
            base, key = _openai_audio()
            name = f"speech.{AUDIO_EXTENSIONS.get(content_type, 'webm')}"
            resp = await client.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data={"model": env("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"), "language": language.split("-")[0]},
                files={"file": (name, audio, content_type)},
            )
            if not resp.is_success:
                raise _failed("OpenAI", resp, resp.content)
            return {"text": str(resp.json().get("text", "")).strip()}
        if provider == "deepgram":
            resp = await client.post(
                "https://api.deepgram.com/v1/listen",
                params={"model": env("DEEPGRAM_MODEL", "nova-3"), "smart_format": "true", "language": language},
                headers={"Authorization": f"Token {require('DEEPGRAM_API_KEY')}", "Content-Type": content_type},
                content=audio,
            )
            if not resp.is_success:
                raise _failed("Deepgram", resp, resp.content)
            alternatives = resp.json()["results"]["channels"][0]["alternatives"]
            return {"text": str(alternatives[0].get("transcript", "")).strip() if alternatives else ""}
    raise HTTPException(400, f"STT_PROVIDER is {provider}: speech is recognised in the browser, not here.")


@router.get("/tts")
async def text_to_speech(text: str) -> StreamingResponse:
    """GET, so an <audio> element can play the stream directly (the access code, if any, goes in ?code=)."""
    text = text.strip()[:1000]
    if not text:
        raise HTTPException(422, "Give the text to speak.")
    provider = tts_provider()
    if provider == "openai":
        base, key = _openai_audio()
        body: dict[str, Any] = {"model": env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"), "voice": env("OPENAI_TTS_VOICE", "alloy"), "input": text, "response_format": "mp3"}
        if instructions := env("OPENAI_TTS_INSTRUCTIONS"):  # gpt-4o-mini-tts only
            body["instructions"] = instructions
        request = httpx.Request("POST", f"{base}/audio/speech", headers={"Authorization": f"Bearer {key}"}, json=body)
    elif provider == "elevenlabs":
        voice_id = env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        request = httpx.Request(
            "POST",
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": require("ELEVENLABS_API_KEY")},
            json={"text": text, "model_id": env("ELEVENLABS_MODEL", "eleven_flash_v2_5")},
        )
    else:
        raise HTTPException(400, f"TTS_PROVIDER is {provider}: speech is made in the browser, not here.")

    client = httpx.AsyncClient(timeout=30)
    resp = await client.send(request, stream=True)
    if not resp.is_success:
        body_bytes = await resp.aread()
        await resp.aclose()
        await client.aclose()
        raise _failed(provider, resp, body_bytes)

    async def audio() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(audio(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
