import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use GPT-OSS-120B as the primary model
# Based on your screenshot, this is the exact model name
PRIMARY_MODEL = "gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-3.5-turbo"  # Fallback if needed


def get_api_key():
    """Get API key from Streamlit secrets."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            key = st.secrets["OPENROUTER_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass
    
    if "openrouter_api_key" in st.session_state and st.session_state.openrouter_api_key:
        return st.session_state.openrouter_api_key
    
    return os.getenv("OPENROUTER_API_KEY")


def call_llm(prompt, system="", max_tokens=2000, temperature=0.7):
    """Call OpenRouter API with GPT-OSS-120B model."""
    
    api_key = get_api_key()
    
    if not api_key:
        return "⚠️ OpenRouter API key not found. Please add to .streamlit/secrets.toml"

    # Try GPT-OSS-120B first
    for attempt, model in enumerate([PRIMARY_MODEL, FALLBACK_MODEL]):
        
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
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            
            elif response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenRouter API key."
                
            elif response.status_code == 429:
                if attempt == 0:
                    continue
                return "⚠️ Rate limit exceeded. Please try again in a few moments."
                
            elif response.status_code == 404 and attempt == 0:
                # Try fallback model
                continue
                
            else:
                if attempt == 0:
                    continue
                return f"⚠️ API error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            if attempt == 0:
                continue
            return "⏰ Request timeout. Please try again."
            
        except requests.exceptions.ConnectionError:
            return "🌐 Connection error. Please check your internet connection."
            
        except Exception as e:
            if attempt == 0:
                continue
            return f"❌ Error: {str(e)[:100]}"
    
    return "⚠️ Unable to generate response. Please try again later."


def get_active_model():
    """Return the active model name for display."""
    return "🧠 GPT-OSS-120B"


def test_api_connection():
    """Test if API key works with GPT-OSS-120B."""
    api_key = get_api_key()
    if not api_key:
        return False, "No API key found"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": PRIMARY_MODEL,
        "messages": [{"role": "user", "content": "Say 'OK'"}],
        "max_tokens": 10,
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return True, "GPT-OSS-120B is working!"
        elif response.status_code == 401:
            return False, "Invalid API key"
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"
