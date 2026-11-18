import os
import asyncio
import time
from pathlib import Path
from typing import Any, Optional, Callable

import httpx
from types import SimpleNamespace
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from loguru import logger


# Global async clients cache for key (base_url, api_key)
_ASYNC_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}


def get_client(base_url: str, api_key_file, api_key_env_var) -> AsyncOpenAI:
    """Get or create cached async OpenAI client.

    Args:
        base_url: API base URL
        api_key_file: Path to API key file
        api_key_env_var: Environment variable name for API key fallback

    Returns:
        Cached AsyncOpenAI client instance
    """
    key_path = Path(api_key_file)
    if not key_path.exists():
        api_key = os.getenv(api_key_env_var)
        if api_key is None:
            raise ValueError(
                f"API key file {key_path} not found and environment variable {api_key_env_var} is not set"
            )
    else:
        if not key_path.is_file():
            raise ValueError(f"API key file {key_path} is not a file")
        api_key = key_path.read_text(encoding="utf-8").strip()
    if len(api_key) == 0:
        raise ValueError("API key is empty")
    if len(base_url) == 0:
        raise ValueError("Base URL is empty")
    cache_key = (base_url, api_key)
    if cache_key not in _ASYNC_CLIENTS:
        _ASYNC_CLIENTS[cache_key] = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=1)
    return _ASYNC_CLIENTS[cache_key]


class _UsageAccumulator:
    """Accumulator for API usage stats across multiple calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.call_count = 0
        self.timeout_count = 0
        self.retry_count = 0

    def add_usage(self, usage: dict | None):
        if usage is None:
            return
        self.call_count += 1
        self.total_cost += usage.get("cost", 0) or 0
        self.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
        self.total_completion_tokens += usage.get("completion_tokens", 0) or 0
        self.total_cached_tokens += usage.get("cached_tokens", 0) or 0

    def summary(self) -> dict:
        return {
            "call_count": self.call_count,
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
        }


class Grader:
    """Base class for LLM-based graders with async-only design.

    Provides common functionality:
    - Async client caching and management via get_client()
    - Async retry logic with exponential backoff
    - Response validation
    - Message building with cache_control support

    Subclasses should implement grader-specific logic like:
    - System prompts
    - Response parsing
    - Batch processing with async methods
    """

    def __init__(
        self,
        grader_model_id: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_file: str = "openrouter_api_key.txt",
        api_key_env_var: str = "OPENROUTER_API_KEY",
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        """Initialize grader with model and API configuration.

        Args:
            grader_model_id: Model identifier for the grading LLM
            base_url: API base URL
            api_key_file: Path to API key file
            api_key_env_var: Environment variable name for API key fallback
            max_retries: Maximum number of retry attempts for API calls
            timeout: Per-request timeout in seconds (default 60s — empirical
                grading call latency is bimodal: <30s normal or stuck at 600s)
        """
        if not isinstance(grader_model_id, str) or len(grader_model_id.strip()) == 0:
            raise ValueError("grader_model_id must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.startswith("http"):
            raise ValueError("base_url must be a valid HTTP(S) URL")
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")

        self.grader_model_id = grader_model_id
        self.base_url = base_url
        self.max_retries = max_retries
        self._timeout = timeout
        self._usage_stats = _UsageAccumulator()

        self._client = get_client(base_url, api_key_file, api_key_env_var)

    def _log_usage(self, completion: Any) -> dict | None:
        """Log token usage and caching info from an API completion.

        Returns a dict with usage details, or None if unavailable.
        """
        usage = getattr(completion, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_cost = getattr(usage, "cost", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = 0
        if prompt_details:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
        cost_str = f"${total_cost:.6f}" if total_cost is not None else "n/a"
        logger.debug(
            f"[grader/{self.grader_model_id}] usage: {prompt_tokens:,} in "
            f"({cached_tokens:,} cached, {cache_pct:.0f}% hit) "
            f"+ {completion_tokens:,} out | cost: {cost_str}"
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cost": total_cost,
        }

    @staticmethod
    def _extract_retry_after(error: APIStatusError) -> float | None:
        """Extract Retry-After header from an API error, if present."""
        response = getattr(error, "response", None)
        if response is None:
            return None
        retry_after = response.headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            return None

    def _validate_response(self, completion: Any) -> None:
        """Validate API response structure.

        Args:
            completion: OpenAI API completion object

        Raises:
            RuntimeError: If response is missing expected fields
        """
        if (
            not getattr(completion, "choices", None)
            or len(completion.choices) == 0
            or completion.choices[0].message is None
        ):
            raise RuntimeError("Empty or invalid response from API")

    def _build_messages(
        self, system_prompt: str, user_prompt: str, use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Build message list with optional cache_control for system prompt.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            use_cache: Whether to enable cache_control for system prompt

        Returns:
            List of message dicts formatted for OpenAI API
        """
        if not isinstance(system_prompt, str) or len(system_prompt.strip()) == 0:
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(user_prompt, str) or len(user_prompt.strip()) == 0:
            raise ValueError("user_prompt must be a non-empty string")

        if use_cache:
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": user_prompt},
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

    async def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        parse_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Make async API call with retry logic and exponential backoff.

        Args:
            messages: Message list for API call
            max_tokens: Maximum tokens in response
            temperature: Optional temperature parameter
            parse_fn: Optional parsing/validation function applied to completion.
                     If provided, parsing errors will trigger retries.
                     Function signature: (completion) -> parsed_result
            **kwargs: Additional parameters for API call

        Returns:
            Completion object (if parse_fn is None) or parsed result (if parse_fn provided)

        Raises:
            Exception: Re-raises last exception after all retries exhausted
        """
        call_params = {
            "model": self.grader_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if temperature is not None:
            call_params["temperature"] = temperature
        # Direct OpenAI calls to gpt-5 reasoning models reject `max_tokens` (it is `max_completion_tokens`
        # there) and any non-default temperature. OpenRouter translates the first and drops the second, so
        # mirroring that here keeps the same requests whichever route serves the model.
        if "api.openai.com" in str(self._client.base_url) and self.grader_model_id.startswith("gpt-5"):
            call_params["max_completion_tokens"] = call_params.pop("max_tokens")
            call_params.pop("temperature", None)

        t_call_start = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                completion = await self._client.chat.completions.create(
                    **call_params, timeout=self._timeout
                )
                elapsed = time.perf_counter() - t0
                self._validate_response(completion)
                usage_info = self._log_usage(completion)
                self._usage_stats.add_usage(usage_info)

                if attempt > 0:
                    total_elapsed = time.perf_counter() - t_call_start
                    logger.info(
                        f"[grader] Succeeded after {attempt} retry(s) "
                        f"({elapsed:.1f}s this attempt, {total_elapsed:.1f}s total)"
                    )
                else:
                    logger.debug(f"[grader] Call completed in {elapsed:.1f}s")

                # If parse_fn provided, call it inside retry loop
                # This means parsing errors trigger retries!
                if parse_fn is not None:
                    return parse_fn(completion)
                else:
                    return completion
            except RateLimitError as e:
                retry_after = self._extract_retry_after(e)
                backoff = retry_after if retry_after else 10.0 * (2 ** attempt)
                self._usage_stats.retry_count += 1
                logger.warning(
                    f"[grader] 429 Rate Limited (attempt {attempt + 1}/{self.max_retries}), "
                    f"backing off {backoff:.0f}s: {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(backoff)
            except (APITimeoutError, APIConnectionError) as e:
                elapsed = time.perf_counter() - t0
                backoff = 3.0 * (2 ** attempt)
                if isinstance(e, APITimeoutError):
                    self._usage_stats.timeout_count += 1
                self._usage_stats.retry_count += 1
                logger.warning(
                    f"[grader] {type(e).__name__} after {elapsed:.0f}s "
                    f"(attempt {attempt + 1}/{self.max_retries}), "
                    f"backing off {backoff:.0f}s"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(backoff)
            except APIStatusError as e:
                if e.status_code >= 500:
                    backoff = 3.0 * (2 ** attempt)
                    logger.warning(
                        f"[grader] Server error {e.status_code} "
                        f"(attempt {attempt + 1}/{self.max_retries}), "
                        f"backing off {backoff:.0f}s: {e}"
                    )
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(backoff)
                elif (e.status_code == 400 and "output limit was reached" in str(e)
                      and "api.openai.com" in str(self._client.base_url)):
                    # Direct OpenAI rejects a reply whose reasoning used up the output limit, where
                    # OpenRouter returns it with empty content. Return the empty reply, which callers
                    # already treat as an unparseable (UNKNOWN) answer.
                    logger.debug("[grader] output limit reached (direct OpenAI): treating as empty reply")
                    empty = SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")],
                        usage=None,
                    )
                    return parse_fn(empty) if parse_fn is not None else empty
                else:
                    logger.error(f"[grader] Permanent API error {e.status_code}: {e}")
                    raise
            except Exception as e:
                logger.error(
                    f"[grader] Error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
