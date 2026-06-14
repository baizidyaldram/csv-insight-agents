import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Reliable free models - ordered by stability
RELIABLE_MODELS = [
    "microsoft/phi-3-mini-128k-instruct:free",  # Most reliable
    "mistralai/mistral-7b-instruct:free",       # Very stable
    "google/gemma-2-9b-it:free",                # Good fallback
    "meta-llama/llama-3-8b-instruct:free",      # Another option
]


def get_api_key():
    """Get API key from Streamlit secrets."""
    try:
        # Try to get from secrets
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    
    # Fallback to session state
    if "openrouter_api_key" in st.session_state and st.session_state.openrouter_api_key:
        return st.session_state.openrouter_api_key
    
    # Fallback to environment
    return os.getenv("OPENROUTER_API_KEY")


def call_llm(prompt, system="", max_tokens=2000, temperature=0.7):
    """Call OpenRouter API with each model tried only once."""
    
    api_key = get_api_key()
    
    if not api_key:
        return "⚠️ OpenRouter API key not found. Please add it to .streamlit/secrets.toml"

    # Try each model once, in order
    for model_index, model in enumerate(RELIABLE_MODELS):
        
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
            "temperature": temperature,
        }

        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
                
            elif response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenRouter API key."
                
            elif response.status_code == 429:
                # Rate limit - try next model
                continue
                
            elif response.status_code == 404:
                # Model not available - try next model
                continue
                
            else:
                # Other error - try next model
                continue
                
        except requests.exceptions.Timeout:
            # Timeout - try next model
            continue
        except requests.exceptions.ConnectionError:
            return "🌐 Connection error. Please check your internet connection."
        except Exception:
            # Other error - try next model
            continue
    
    # If we've tried all models and none worked
    return "⚠️ All AI models are currently unavailable. Please try again in a few moments."


def get_active_model():
    """Return the active model name for display."""
    return "🧠 AI Model"


def is_api_available():
    """Check if API key is configured."""
    return get_api_key() is not None
