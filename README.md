<div align="center">

# 🌟 Celeste AI Client

### One Interface, All AI Providers - Unified API for Seamless AI Integration

[![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Providers](https://img.shields.io/badge/Providers-6+-orange?style=for-the-badge&logo=openai&logoColor=white)](#-supported-providers)
[![Models](https://img.shields.io/badge/Models-40+-purple?style=for-the-badge&logo=huggingface&logoColor=white)](#-supported-models)

[![Demo](https://img.shields.io/badge/🚀_Try_Demo-Streamlit-FF4B4B?style=for-the-badge)](example.py)
[![Documentation](https://img.shields.io/badge/📚_Docs-Coming_Soon-blue?style=for-the-badge)](#)

</div>

---

## 🎯 Why Celeste?

<div align="center">
  <table>
    <tr>
      <td align="center">🔌<br><b>Unified API</b><br>One interface for all providers</td>
      <td align="center">🏠<br><b>Local & Cloud</b><br>Run locally or in the cloud</td>
      <td align="center">⚡<br><b>Async First</b><br>Built for performance</td>
      <td align="center">🔄<br><b>Streaming</b><br>Real-time responses</td>
    </tr>
  </table>
</div>

## 🚀 Quick Start

```python
# Install
pip install celeste-client  # Coming soon to PyPI

# Use any AI provider with the same interface
from celeste_client import create_client, Provider

# Cloud providers
client = create_client(Provider.OPENAI, model="gpt-4o-mini")
client = create_client(Provider.ANTHROPIC, model="claude-3-7-sonnet")

# Local models (no API key needed!)
client = create_client(Provider.OLLAMA, model="llama3.2")

# Generate content
response = await client.generate_content("Explain quantum computing")
print(response.content)

# Get usage details
print(response.usage)  # Token usage and costs
```

## 📦 Installation

<details open>
<summary><b>Using UV (Recommended)</b></summary>

```bash
git clone https://github.com/yourusername/celeste-client
cd celeste-client
uv sync
```
</details>

<details>
<summary><b>Using pip</b></summary>

```bash
git clone https://github.com/yourusername/celeste-client
cd celeste-client
pip install -e .
```
</details>

## 🔧 Configuration

### 1️⃣ Create your environment file
```bash
cp .env.example .env
```

### 2️⃣ Add your API keys

<details>
<summary><b>🔑 API Key Setup</b></summary>

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| 🌈 **Gemini** | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| 🤖 **OpenAI** | `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/api-keys) |
| 🌊 **Mistral** | `MISTRAL_API_KEY` | [Mistral Console](https://console.mistral.ai/) |
| 🎭 **Anthropic** | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) |
| 🤗 **Hugging Face** | `HUGGINGFACE_TOKEN` | [HF Settings](https://huggingface.co/settings/tokens) |
| 🦙 **Ollama** | *No key needed!* | [Install Ollama](https://ollama.com/download) |

</details>

## 🎨 Supported Providers

<div align="center">

| Provider | Models | Streaming | Structured Output | Local | Free Tier |
|----------|--------|-----------|-------------------|--------|-----------|
| 🌈 **Google Gemini** | 3 | ✅ | ✅ | ❌ | ✅ |
| 🤖 **OpenAI** | 3 | ✅ | ✅ | ❌ | ❌ |
| 🌊 **Mistral AI** | 4 | ✅ | 🔜 Coming Soon | ❌ | ✅ |
| 🎭 **Anthropic** | 3 | ✅ | 🔜 Coming Soon | ❌ | ❌ |
| 🤗 **Hugging Face** | 7 | ✅ | 🔜 Coming Soon | ❌ | ✅ |
| 🦙 **Ollama** | 20+ | ✅ | 🔜 Coming Soon | ✅ | ✅ |

</div>

## 📊 Supported Models

<details>
<summary><b>View All Models</b></summary>

### 🌈 Google Gemini
- `gemini-2.5-flash-lite-preview-06-17` - Ultra-fast responses
- `gemini-2.5-flash` - Balanced performance
- `gemini-2.5-pro` - Maximum capability

### 🤖 OpenAI
- `o3-2025-04-16` - Latest O3 model
- `o4-mini-2025-04-16` - Cost-effective
- `gpt-4.1-2025-04-14` - Advanced reasoning

### 🌊 Mistral AI
- `mistral-small-latest` - Fast & efficient
- `mistral-medium-latest` - Balanced
- `mistral-large-latest` - High performance
- `codestral-latest` - Code specialist

### 🎭 Anthropic Claude
- `claude-3-7-sonnet-20250219` - Claude 3.7
- `claude-sonnet-4-20250514` - Claude 4 Sonnet
- `claude-opus-4-20250514` - Claude 4 Opus

### 🤗 Hugging Face
- `google/gemma-2-2b-it` - Lightweight
- `meta-llama/Meta-Llama-3.1-8B-Instruct` - Llama 3.1
- `microsoft/phi-4` - Microsoft Phi-4
- `Qwen/Qwen2.5-7B-Instruct-1M` - 1M context window
- `deepseek-ai/DeepSeek-R1` - DeepSeek reasoning
- [View more...](#)

### 🦙 Ollama (Local)
Popular models (pull with `ollama pull <model>`):
- `llama3.2` - Latest Llama
- `mistral` - Mistral 7B
- `mixtral:8x7b` - MoE model
- `phi3` - Microsoft Phi-3
- `deepseek-r1` - DeepSeek reasoning
- `codellama` - Code generation
- [View all models](https://ollama.com/library)

</details>

## 💻 Usage Examples

### 🔥 Streaming Responses
```python
async for chunk in client.stream_generate_content("Write a haiku about programming"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 🏠 Local Models with Ollama
```python
# No API key needed!
client = create_client(Provider.OLLAMA, model="llama3.2")

# Custom host
client = create_client(Provider.OLLAMA, model="llama3.2", 
                      host="http://192.168.1.100:11434")
```

### 🎯 Provider Comparison
```python
from celeste_client import create_client, Provider

providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.MISTRAL]
prompt = "Explain quantum entanglement in one sentence"

for provider in providers:
    client = create_client(provider)
    response = await client.generate_content(prompt)
    print(f"{provider.value}: {response.content}")
    print(f"Usage: {response.usage}")
```

### 🎯 Structured Output with Pydantic (NEW!)

Generate structured data with type safety using Pydantic models:

```python
from pydantic import BaseModel
from celeste_client import create_client, Provider, GeminiModel

# Define your data structure
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Single object generation
client = create_client(Provider.GOOGLE, model=GeminiModel.FLASH)
response = await client.generate_content(
    "Generate a person profile for a software engineer",
    response_schema=Person
)
print(response.content)  # Person(name='Alice Chen', age=28, occupation='Senior Software Engineer')
print(response.content.name)  # Direct access to fields: 'Alice Chen'

# List generation
response = await client.generate_content(
    "Generate a list of 5 team members",
    response_schema=list[Person]
)
for person in response.content:
    print(f"{person.name} - {person.occupation}")
```

**Currently supported:** 🌈 Gemini, 🤖 OpenAI (other providers coming soon!)

## 🎮 Interactive Demo

Try our Streamlit demo with all providers:

```bash
uv run streamlit run example.py
```

<div align="center">
  <img src="assets/demo-screenshot.png" width="600" alt="Celeste Demo">
</div>

## 🗺️ Roadmap

### Celeste-Client Next Steps
- [x] 📝 **Use Types** - Implement AIPrompt and AIResponse types
- [x] 📊 **Add Metadata** - Generation time and token usage tracking
- [x] 🎯 **Structured Output** - Pydantic model support (Gemini ✅, OpenAI ✅)
- [ ] 🔄 **Structured Output for All** - Extend to OpenAI, Anthropic, Mistral
- [ ] 📚 **Sphinx Documentation** - Comprehensive API documentation
- [ ] 🧪 **Unit Tests** - Achieve 80% test coverage
- [ ] 🛡️ **Error Handling** - Robust error handling and retry logic
- [ ] 📦 **PyPI Package** - Easy installation

### Celeste Ecosystem

| Package | Description | Status |
|---------|-------------|--------|
| 💬 **celeste-conversations** | Multi-turn conversations with memory management | 🔄 In Progress |
| 🌐 **celeste-web-agent** | Web browsing and automation capabilities | 📋 Backlog |
| 🎨 **celeste-image-generation** | Image generation across providers | ✅ Done |
| ✏️ **celeste-image-edit** | Image editing and manipulation | ✅ Done |
| 🎬 **celeste-video-generation** | Video generation and editing | 📋 Backlog |
| 📊 **celeste-presentation-intelligence** | PowerPoint and presentation analysis | 📋 Backlog |
| 📄 **celeste-document-intelligence** | PDF and document processing | ✅ Done |
| 🔢 **celeste-embeddings** | Text embeddings across providers | ✅ Done |
| 📈 **celeste-table-intelligence** | Excel, CSV, and Parquet analysis | 📋 Backlog |
| 🖼️ **celeste-image-intelligence** | Image analysis and understanding | 📋 Backlog |
| 🎥 **celeste-video-intelligence** | Video analysis and understanding | 📋 Backlog |
| 🚀 **And many more...** | Expanding ecosystem of AI tools | 🔮 Future |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by the Celeste Team
  
  <a href="#-celeste-ai-client">⬆ Back to Top</a>
</div>