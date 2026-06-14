import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model settings - reliable free models
# Order matters - put most reliable first
RELIABLE_MODELS = [
    "microsoft/phi-3-mini-128k-instruct:free",  # Most reliable
    "mistralai/mistral-7b-instruct:free",       # Very stable
    "google/gemma-2-9b-it:free",                # Good fallback
    "meta-llama/llama-3-8b-instruct:free",      # Another option
]

# Track attempted models per session to avoid infinite loops
if "attempted_models" not in st.session_state:
    st.session_state.attempted_models = {}


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


def call_llm(prompt, system="", max_retries=3, max_tokens=2000, temperature=0.7):
    """Call OpenRouter API with limited retries and no infinite loops."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add your OpenRouter API key in the sidebar to enable AI features."

    # Track attempts for this specific prompt (hash it to avoid memory issues)
    prompt_hash = hash(prompt[:100])  # Use first 100 chars as identifier
    
    if prompt_hash not in st.session_state.attempted_models:
        st.session_state.attempted_models[prompt_hash] = []
    
    attempted = st.session_state.attempted_models[prompt_hash]
    
    # Try models in order, but only once each
    for model in RELIABLE_MODELS:
        if model in attempted:
            continue  # Already tried this model
            
        attempted.append(model)
        
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
                # Clear the attempted models for this prompt on success
                st.session_state.attempted_models[prompt_hash] = []
                return data["choices"][0]["message"]["content"].strip()
                
            elif response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenRouter API key."
                
            elif response.status_code == 429:
                # Rate limit - wait a bit then continue to next model
                import time
                time.sleep(2)
                continue
                
            elif response.status_code == 404:
                # Model not available - try next
                continue
                
            else:
                # Other error - try next model
                continue
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            return "🌐 Connection error. Please check your internet connection."
        except Exception:
            continue
    
    # If we get here, all models failed
    # Clear attempted models for next time
    st.session_state.attempted_models[prompt_hash] = []
    return "⚠️ All AI models are currently unavailable. Please try again in a few moments."


def get_active_model():
    """Return the primary active model name for display."""
    return "🧠 AI Model (Auto-select)"


def test_models():
    """Test which models are working - call this once to cache results."""
    if "working_models" in st.session_state:
        return st.session_state.working_models
    
    api_key = get_api_key()
    if not api_key:
        return []
    
    working = []
    for model in RELIABLE_MODELS:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5,
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                working.append(model)
        except:
            pass
    
    st.session_state.working_models = working
    return working
