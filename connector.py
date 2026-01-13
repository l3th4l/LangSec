import json
from typing import Any, Dict, List, Optional, Union

import requests


class OpenRouterChatConnector:
    """
    Minimal connector for OpenRouter chat completions.

    - Establishes a session (connection pooling via requests.Session).
    - Maintains message history internally.
    - Optionally enables reasoning and preserves reasoning_details unmodified.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        enable_reasoning: bool = True,
        timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.enable_reasoning = enable_reasoning
        self.timeout = timeout

        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            self.headers.update(extra_headers)

        self.messages: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def add_message(self, role: str, content: str, **extra_fields: Any) -> None:
        """
        Add a message to history. Extra fields are kept (e.g., reasoning_details).
        """
        msg: Dict[str, Any] = {"role": role, "content": content}
        if extra_fields:
            msg.update(extra_fields)
        self.messages.append(msg)

    def _build_payload(self, override_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": override_messages if override_messages is not None else self.messages,
        }
        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            url=self.base_url,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        # Helpful error on non-2xx responses
        if not r.ok:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
        return r.json()

    def send(
        self,
        user_content: str,
        *,
        preserve_reasoning_details: bool = True,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """
        Add a user message, call the API, append the assistant reply to history,
        and return a structured result.

        Returns dict with:
          - content: assistant text
          - reasoning_details: (optional) whatever the API returned
          - raw: full JSON response
          - assistant_message: the message object returned by the API
        """
        self.add_message(user_role, user_content)

        payload = self._build_payload()
        raw = self._post(payload)

        # OpenRouter-style response parsing
        assistant_msg = raw["choices"][0]["message"]
        content = assistant_msg.get("content")

        # Preserve reasoning_details exactly as received, if present
        if preserve_reasoning_details and "reasoning_details" in assistant_msg:
            self.add_message(
                "assistant",
                content if content is not None else "",
                reasoning_details=assistant_msg.get("reasoning_details"),
            )
        else:
            self.add_message("assistant", content if content is not None else "")

        return {
            "content": content,
            "reasoning_details": assistant_msg.get("reasoning_details"),
            "assistant_message": assistant_msg,
            "raw": raw,
        }

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "OpenRouterChatConnector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ----------------------------
# Example usage (matches your flow)
# ----------------------------
if __name__ == "__main__":
    API_KEY = "<API_KEY>"
    MODEL = "xiaomi/mimo-v2-flash:free"

    with OpenRouterChatConnector(api_key=API_KEY, model=MODEL, enable_reasoning=True) as conn:
        out1 = conn.send("How many r's are in the word 'strawberry'?")
        print("Assistant:", out1["content"])

        out2 = conn.send("Are you sure? Think carefully.")
        print("Assistant:", out2["content"])
