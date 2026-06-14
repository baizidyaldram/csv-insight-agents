import os
import requests
import streamlit as st

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model settings - prioritizing reliable free models
# GPT-OSS 120B is a powerful open-source model with 120B parameters
PRIMARY_MODEL = "cognitivecomputations/dolphin-mixtral-8x22b:free"
BACKUP_MODELS = [
    "microsoft/phi-3-mini-128k-instruct:free",  # Microsoft's Phi-3 Mini - very capable
    "google/gemma-2-9b-it:free",                 # Google's Gemma 2
    "meta-llama/llama-3-8b-instruct:free",       # Meta's Llama 3
    "mistralai/mistral-7b-instruct:free",        # Mistral 7B
]

# Alternative good free models you can try (swap PRIMARY_MODEL with any of these):
# "openchat/openchat-3.5-1210:free" - Excellent for conversations
# "nousresearch/nous-hermes-2-mixtral-8x7b:free" - Good for complex reasoning
# "huggingfaceh4/zephyr-7b-beta:free" - Good for instruction following
# "teknium/openhermes-2.5-mistral-7b:free" - Good general purpose


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


def get_available_models():
    """Return list of recommended free models."""
    return [PRIMARY_MODEL] + BACKUP_MODELS


def call_llm(prompt, system="", model_index=0, max_tokens=2000, temperature=0.7):
    """Call OpenRouter API with intelligent model fallback and retry logic."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ API key not found. Please add your OpenRouter API key in the sidebar to enable AI features."

    # Select model with fallback cycle
    available_models = [PRIMARY_MODEL] + BACKUP_MODELS
    
    if model_index >= len(available_models):
        # Try a different approach - use a known working model
        model = "microsoft/phi-3-mini-128k-instruct:free"
        retry_msg = "⚠️ Multiple models failed. Using stable fallback model."
    else:
        model = available_models[model_index]

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
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        elif response.status_code == 401:
            return "❌ Invalid API key. Please check your OpenRouter API key."
            
        elif response.status_code == 429:
            # Rate limit - wait and retry with different model
            if model_index < len(available_models) - 1:
                return call_llm(prompt, system, model_index + 1, max_tokens, temperature)
            return "⚠️ Rate limit exceeded. Please try again in a few moments."
            
        elif response.status_code == 404:
            # Model not available - try next model
            if model_index < len(available_models) - 1:
                # Show which model we're switching to
                next_model = available_models[model_index + 1]
                model_name_short = next_model.split('/')[-1].replace(':free', '')
                st.info(f"🔄 Switching to {model_name_short}...")
                return call_llm(prompt, system, model_index + 1, max_tokens, temperature)
            else:
                # Ultimate fallback
                return call_llm(prompt, system, 0, max_tokens, temperature)
                
        else:
            # Other error - try next model
            if model_index < len(available_models) - 1:
                return call_llm(prompt, system, model_index + 1, max_tokens, temperature)
            return f"❌ API error: {response.status_code}\n{response.text[:300]}"
            
    except requests.exceptions.Timeout:
        if model_index < len(available_models) - 1:
            st.warning(f"⏰ Timeout with current model, trying next...")
            return call_llm(prompt, system, model_index + 1, max_tokens, temperature)
        return "⏰ Request timed out. Please try again."
        
    except requests.exceptions.ConnectionError:
        return "🌐 Connection error. Please check your internet connection."
        
    except Exception as e:
        if model_index < len(available_models) - 1:
            return call_llm(prompt, system, model_index + 1, max_tokens, temperature)
        return f"❌ Error: {str(e)}"


def call_llm_with_retry(prompt, system="", max_retries=3, **kwargs):
    """Wrapper function with explicit retry logic for important calls."""
    for attempt in range(max_retries):
        result = call_llm(prompt, system, model_index=attempt, **kwargs)
        
        # Check if result indicates an error that should trigger retry
        error_indicators = ["❌", "⚠️", "Error:", "timeout", "failed"]
        if not any(indicator in result.lower() for indicator in error_indicators):
            return result
            
        if attempt < max_retries - 1:
            st.info(f"Retrying with different model (attempt {attempt + 2}/{max_retries})...")
    
    return result


def get_active_model():
    """Return the primary active model name (short version for display)."""
    model_name = PRIMARY_MODEL.split('/')[-1].replace(':free', '').replace('-instruct', '')
    # Clean up common suffixes
    model_name = model_name.replace('-8x22b', '').replace('-7b', '').replace('-8b', '')
    return f"🧠 {model_name}"


def list_available_free_models():
    """Return a list of known good free models on OpenRouter."""
    return [
        "cognitivecomputations/dolphin-mixtral-8x22b:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "google/gemma-2-9b-it:free",
        "meta-llama/llama-3-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "openchat/openchat-3.5-1210:free",
        "nousresearch/nous-hermes-2-mixtral-8x7b:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "teknium/openhermes-2.5-mistral-7b:free",
    ]


def test_model_availability(model_name: str) -> bool:
    """Quick test if a model is available."""
    api_key = get_api_key()
    if not api_key:
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5,
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
        return response.status_code == 200
    except:
        return False
