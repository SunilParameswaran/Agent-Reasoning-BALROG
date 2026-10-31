"""Gemini's documented OpenAI-compatible transport for the BALROG baseline."""
import os

from openai import OpenAI, RateLimitError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def connect_gemini():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("Set GEMINI_API_KEY in your terminal before using --provider gemini.")
    # The installed SDK retries temporary failures and honours Retry-After headers.
    return OpenAI(api_key=key, base_url=GEMINI_BASE_URL, timeout=60, max_retries=5)


class GeminiClient:
    def __init__(self, sdk, config):
        self.sdk = sdk
        self.config = config

    def generate(self, messages):
        from balrog.client import LLMResponse

        try:
            result = self.sdk.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.config.generate_kwargs.temperature,
                max_tokens=self.config.generate_kwargs.max_tokens,
                reasoning_effort=self.config.generate_kwargs.reasoning_effort,
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "Gemini rate limit/quota exhausted after retries. Check your project's "
                "limits in AI Studio. Increase --request-delay for per-minute limits; "
                "daily quota requires waiting for reset. Completed episode files are "
                "preserved. After quota is available, use --resume RUN_FOLDER to continue."
            ) from exc
        if not result.choices:
            raise RuntimeError("Gemini returned no candidate action; check safety filters and model access.")
        choice = result.choices[0]
        if (not choice.message.content or not choice.message.content.strip()) and not getattr(self, "allow_empty_response", False):
            raise RuntimeError(
                f"Gemini returned no action text (finish_reason={choice.finish_reason}). "
                "If the token limit was reached, increase --max-tokens; "
                "thinking can consume part of the response budget."
            )
        if result.usage is None:
            raise RuntimeError("Gemini returned no token usage; cannot record the experiment accurately.")
        return LLMResponse(
            model_id=self.config.model_id,
            completion=(choice.message.content or "").strip(),
            stop_reason=choice.finish_reason,
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            reasoning=None,
        )
