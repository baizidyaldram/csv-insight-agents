import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free-tier models on OpenRouter
FREE_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nousresearch/hermes-3-llama-3.1-8b:free",
]


def get_api_key() -> str | None:
    """Get API key from env or Streamlit secrets."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
    return key


def call_llm(prompt: str, system: str = "", model_index: int = 0, max_tokens: int = 1500) -> str:
    """Call OpenRouter with fallback across free models."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ OPENROUTER_API_KEY not found. Add it to your .env file or Streamlit secrets for AI features."

    model = FREE_MODELS[model_index % len(FREE_MODELS)]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://csv-insight-agents.streamlit.app",
        "X-Title": "CSV Insight Agents",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Try again."
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429 and model_index + 1 < len(FREE_MODELS):
            return call_llm(prompt, system, model_index + 1, max_tokens)
        return f"❌ API error: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


def get_active_model(index: int = 0) -> str:
    return FREE_MODELS[index % len(FREE_MODELS)]
