import os
import warnings

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY is None:
    warnings.warn("ANTHROPIC_API_KEY not set", stacklevel=2)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    warnings.warn("GOOGLE_API_KEY not set", stacklevel=2)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None:
    warnings.warn("HUGGINGFACE_TOKEN not set", stacklevel=2)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY is None:
    warnings.warn("MISTRAL_API_KEY not set", stacklevel=2)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    warnings.warn("OPENAI_API_KEY not set", stacklevel=2)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
