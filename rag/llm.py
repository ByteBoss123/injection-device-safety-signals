"""LLM clients. AzureChat is the production path (Azure OpenAI chat completions with function calling and
structured outputs). ScriptedChat replays fixed responses for deterministic tests."""
import json, os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _azure_client():
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                       max_retries=int(os.environ.get("AZURE_OPENAI_MAX_RETRIES", "8")))  # backs off on 429


class AzureChat:
    def __init__(self, deployment=None):
        self.client = _azure_client()
        self.deployment = deployment or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    def __call__(self, messages, tools=None, response_schema=None, temperature=0.0):
        kw = {"model": self.deployment, "messages": messages, "temperature": temperature}
        if tools:
            kw["tools"] = tools
        if response_schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": response_schema}
        r = self.client.chat.completions.create(**kw)
        m = r.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}")) for c in (m.tool_calls or [])]
        return ChatResult(m.content, calls, {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens})


def embedder(dimensions=256, deployment=None):
    client = _azure_client()
    dep = deployment or os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

    def embed(texts):
        r = client.embeddings.create(model=dep, input=texts, dimensions=dimensions)
        return [d.embedding for d in r.data]
    return embed


class ScriptedChat:
    """Returns pre-set ChatResults in order; records every request for assertions."""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def __call__(self, messages, tools=None, response_schema=None, temperature=0.0):
        self.requests.append({"messages": messages, "tools": tools, "schema": response_schema})
        step = self.script.pop(0)
        return step(messages) if callable(step) else step
