import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model settings - using reliable models
PRIMARY_MODEL = "mistralai/mistral-7b-instruct:free"
FALLBACK_MODELS = [
    "gryphe/mythomax-l2-13b:free",
    "openai/gpt-3.5-turbo",
]


def get_api_key():
    """Get API key from session state, env, or Streamlit secrets."""
    if "openrouter_api_key" in st.session_state and st.session_state.openrouter_api_key:
        return st.session_state.openrouter_api_key
        
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
    return key


def call_llm(prompt, system="", model_index=0, max_tokens=1500):
    """Call OpenRouter API with better error handling."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add your OpenRouter API key in the sidebar to enable AI features."

    # Select model
    if model_index == 0:
        model = PRIMARY_MODEL
    else:
        idx = (model_index - 1) % len(FALLBACK_MODELS)
        model = FALLBACK_MODELS[idx]

    # Build messages
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
        "temperature": 0.7,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 401:
            return "❌ Invalid API key. Please check your OpenRouter API key."
        elif response.status_code == 429:
            return "⚠️ Rate limit exceeded. Please try again in a few moments."
        elif response.status_code == 404:
            if model_index == 0:
                return call_llm(prompt, system, model_index + 1, max_tokens)
            return f"⚠️ Model {model} not available. Using fallback model."
        elif response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            if model_index < len(FALLBACK_MODELS):
                return call_llm(prompt, system, model_index + 1, max_tokens)
            return f"❌ API error: {response.status_code} - {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "🌐 Connection error. Please check your internet connection."
    except Exception as e:
        if model_index < len(FALLBACK_MODELS):
            return call_llm(prompt, system, model_index + 1, max_tokens)
        return f"❌ Error: {str(e)}"


def get_active_model():
    """Return the active model name."""
    return PRIMARY_MODEL
