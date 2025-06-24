# 🌟 Celeste AI Client

A Python client library for Google's Gemini AI models with support for thinking capabilities.

## Features

- **Multiple Gemini Models**: Support for Gemini 2.5 Flash, Flash Lite, and Pro
- **Thinking Support**: Built-in support for Gemini's thinking capabilities
- **Async/Await**: Full async support for both standard and streaming responses
- **Streamlit Demo**: Interactive web interface for testing

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
# Edit .env and add your Google API key
```

## API Key Setup

Get your Google AI API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and add it to your `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
import asyncio
from celeste_client import create_client

async def main():
    client = create_client("gemini", model="gemini-2.5-flash")
    
    # Standard generation
    response = await client.generate_content("Why is the sky blue?")
    print(response)
    
    # Streaming generation
    async for chunk in client.stream_generate_content("Tell me a story"):
        print(chunk, end="")

asyncio.run(main())
```

### Available Models

- `gemini-2.5-flash-lite-preview-06-17` (Flash Lite)
- `gemini-2.5-flash` (Flash)
- `gemini-2.5-pro` (Pro)

### Streamlit Demo

Run the interactive demo:

```bash
uv run streamlit run example.py
```

## Requirements

- Python >= 3.13
- UV package manager
- Google AI API key

## License

MIT