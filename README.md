# 🌟 Celeste AI Client

A Python client library for multiple AI providers with unified API interface and streaming support.

## Features

- **Multi-Provider Support**: Unified interface for Gemini, OpenAI, Mistral, and Anthropic Claude
- **Streaming Support**: Real-time streaming responses for all providers
- **Async/Await**: Full async support for both standard and streaming responses
- **Interactive Demo**: Streamlit web interface with provider and model selection
- **Type Safety**: Full type hints and enum-based model selection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd celeste-client
```

2. Install dependencies using UV:
```bash
uv sync
```

3. Set up your environment:
```bash
cp .env.example .env
# Edit .env and add your API keys for the providers you want to use
```

## API Key Setup

Add API keys for the providers you want to use in your `.env` file:

```bash
# Gemini
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Mistral
MISTRAL_API_KEY=your_mistral_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Getting API Keys

- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **OpenAI**: [OpenAI API Platform](https://platform.openai.com/api-keys)
- **Mistral**: [Mistral AI Platform](https://console.mistral.ai/)
- **Anthropic**: [Anthropic Console](https://console.anthropic.com/)

## Usage

### Basic Usage

```python
import asyncio
from celeste_client import create_client

async def main():
    # Use any supported provider
    client = create_client("openai", model="gpt-4o-mini")
    # client = create_client("gemini", model="gemini-2.5-flash")
    # client = create_client("mistral", model="mistral-large-latest")
    # client = create_client("anthropic", model="claude-3-7-sonnet-20250219")
    
    # Standard generation
    response = await client.generate_content("Why is the sky blue?")
    print(response)
    
    # Streaming generation
    async for chunk in client.stream_generate_content("Tell me a story"):
        print(chunk, end="")

asyncio.run(main())
```

### Supported Providers & Models

#### Google Gemini
- `gemini-2.5-flash-lite-preview-06-17` (Flash Lite)
- `gemini-2.5-flash` (Flash)
- `gemini-2.5-pro` (Pro)

#### OpenAI
- `o3-2025-04-16` (O3)
- `o4-mini-2025-04-16` (O4 Mini)
- `gpt-4.1-2025-04-14` (GPT-4.1)

#### Mistral AI
- `mistral-small-latest` (Small)
- `mistral-medium-latest` (Medium)
- `mistral-large-latest` (Large)
- `codestral-latest` (Codestral)

#### Anthropic Claude
- `claude-3-7-sonnet-20250219` (Claude 3.7 Sonnet)
- `claude-sonnet-4-20250514` (Claude 4 Sonnet)
- `claude-opus-4-20250514` (Claude 4 Opus)

### Streamlit Demo

Run the interactive demo:

```bash
uv run streamlit run example.py
```

## Requirements

- Python >= 3.13
- UV package manager
- API keys for desired providers (see API Key Setup section)

## License

MIT