# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Celeste Client is a unified AI client library that provides a single interface for interacting with multiple AI providers (OpenAI, Anthropic, Google, Mistral, HuggingFace, Ollama, and Transformers). It's part of the larger Celeste ecosystem and focuses specifically on text generation capabilities.

## Architecture

### Core Components
- **Factory Pattern**: `create_client()` function in `src/celeste_client/__init__.py` dynamically imports and instantiates provider clients
- **Provider Mapping**: `src/celeste_client/mapping.py` defines the relationship between providers and their implementation modules
- **Base Client**: All provider clients inherit from `BaseClient` in the celeste-core dependency
- **Provider Implementations**: Located in `src/celeste_client/providers/`, each provider has its own client class

### Key Design Patterns
- **Dynamic Module Loading**: Uses `importlib.import_module()` to load provider clients on demand
- **Capability Validation**: Each client validates that models support `TEXT_GENERATION` capability
- **Unified Response Format**: All providers return `AIResponse` objects with consistent structure
- **Streaming Support**: All providers implement both sync and async streaming via `AsyncIterator[AIResponse]`

### Dependencies
- **celeste-core**: Provides base classes, enums, validation, and model catalog
- **Provider SDKs**: Each provider uses its official SDK (anthropic, google-genai, openai, etc.)
- **Settings Management**: Environment variables handled through celeste-core settings

## Development Commands

### Setup and Installation
```bash
uv sync                           # Install all dependencies
uv pip install -e . --group dev  # Install in editable mode with dev dependencies
```

### Development Tools
```bash
pre-commit install               # Install pre-commit hooks
pre-commit run --files <files>   # Run linting and formatting on specific files
uv run streamlit run example.py  # Run interactive demo with all providers
```

### Code Quality
- **Ruff**: Handles linting and formatting (configured in pre-commit)
- **mypy**: Type checking (configured in pre-commit)
- **Pre-commit hooks**: Run ruff, ruff-format, mypy, and prevent commits to main/master

## Adding New Providers

### Implementation Steps
1. **Create Provider Client**: `src/celeste_client/providers/<provider>.py`
   - Inherit from `BaseClient`
   - Implement `generate_content()` and `stream_generate_content()` methods
   - Call `super().__init__(model=model, provider=Provider.X, **kwargs)`

2. **Wire Provider**: Edit `src/celeste_client/mapping.py`
   - Add entry to `PROVIDER_MAPPING` dictionary
   - Format: `Provider.NAME: (".providers.module", "ClassName")`

3. **Register Models**: Add models to celeste-core catalog
   - Edit `celeste-core/src/celeste_core/models/catalog.py`
   - Include `capability=Capability.TEXT_GENERATION`

4. **Dependencies**: Add provider SDK to `pyproject.toml` if needed

### Implementation Requirements
- **Error Handling**: Use BaseClient for capability validation
- **Streaming**: Stream responses must set `is_stream_chunk=True` in metadata
- **Kwargs Passthrough**: Accept and pass through `**kwargs` for provider-specific options
- **Content Filtering**: Only yield streaming chunks that contain actual content

## Key Files and Locations

### Core Implementation
- `src/celeste_client/__init__.py` - Main factory function and exports
- `src/celeste_client/mapping.py` - Provider to implementation mapping
- `src/celeste_client/providers/` - Individual provider implementations

### Configuration and Setup
- `pyproject.toml` - Project dependencies and UV configuration
- `.pre-commit-config.yaml` - Code quality automation
- `example.py` - Streamlit demo application

### Documentation
- `README.md` - User-facing documentation with supported providers/models
- `CONTRIBUTING.md` - Contributor guidelines and provider addition process

## Testing and Validation

### Manual Testing
```bash
uv run streamlit run example.py  # Interactive testing with UI
```

### Provider Testing
- Use the Streamlit demo to test both `generate_content` and `stream_generate_content`
- Verify proper error handling for missing API keys
- Test model capability validation

## Environment Configuration

### Required API Keys (as needed)
- `GOOGLE_API_KEY` - Google Gemini models
- `OPENAI_API_KEY` - OpenAI models
- `ANTHROPIC_API_KEY` - Claude models
- `MISTRAL_API_KEY` - Mistral models
- `HUGGINGFACE_TOKEN` - HuggingFace models
- No key needed for Ollama (local models)

### Model Catalog
Models are centrally managed in the celeste-core dependency. The client validates that requested models support text generation capability before instantiating providers.
