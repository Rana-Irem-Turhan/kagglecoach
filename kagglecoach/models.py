from __future__ import annotations
from typing import Any
import os

FOUNDRY_BASE_URL = "http://127.0.0.1:60458"


class ModelClient:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._embedder = None

    def _use_azure(self) -> bool:
        return self._settings.azure.enabled

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _openai_client(self):
        import openai
        if self._use_azure():
            return openai.AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                api_version=self._settings.azure.api_version,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            )
        return openai.OpenAI(
            base_url=FOUNDRY_BASE_URL + "/v1",
            api_key="foundry-local",
        )

    def embed(self, texts):
        embedder = self._get_embedder()
        vectors = embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def chat(self, messages_or_system, user_prompt=None, temperature=0.3):
        if user_prompt is not None:
           messages = [
             {"role": "system", "content": messages_or_system},
             {"role": "user", "content": user_prompt},
           ]
        else:
           messages = messages_or_system
        client = self._openai_client()
        model = self._settings.chat_model if self._use_azure() else "qwen2.5-1.5b-instruct-cuda-gpu:4"
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    @property
    def active_mode(self) -> str:
        return "azure" if self._use_azure() else "local"
    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
    def warmup(self) -> None:
        self._get_embedder()
