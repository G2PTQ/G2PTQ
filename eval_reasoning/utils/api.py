"""Readiness probe for the OpenAI-compatible endpoint used by EvalScope."""

from __future__ import annotations

import time
import urllib.error
import urllib.request


# How long one probe may hang, kept separate from the polling interval so that a
# short interval does not also cut every request short.
PROBE_TIMEOUT = 10.0


def wait_for_api(api_url: str, timeout: float, interval: float) -> None:
    """Poll ``/models`` until the server answers, or raise once ``timeout`` passes.

    A vLLM server takes minutes to load a large checkpoint, so a single probe at
    startup would fail on a service that is merely still loading. Polling lets
    the driver be launched alongside the server instead of after it.
    """
    url = api_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as response:
                if response.status == 200:
                    print(f"vLLM API is ready at {url}")
                    return
                reason = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            reason = str(exc)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"vLLM API at {url} was not ready within {timeout:.0f}s "
                f"({attempt} attempts); last error: {reason}"
            )
        print(
            f"waiting for vLLM at {url}: {reason} "
            f"(retrying in {interval:.0f}s, {remaining:.0f}s left)",
            flush=True,
        )
        time.sleep(min(interval, remaining))
