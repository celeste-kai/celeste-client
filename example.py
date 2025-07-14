import asyncio
from typing import List, Tuple

import streamlit as st

from celeste_client import create_client
from celeste_client.core.enums import (
    AnthropicModel,
    GoogleModel,
    HuggingFaceModel,
    MistralModel,
    OllamaModel,
    OpenAIModel,
    Provider,
)
from celeste_client.core.types import AIResponse

st.set_page_config(page_title="Celeste AI", page_icon="🌟", layout="wide")

st.title("🌟 Celeste AI Client")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    def format_provider(x: str) -> str:
        return x.title()

    selected_provider = st.selectbox(
        "Provider:",
        options=[provider.value for provider in Provider],
        format_func=format_provider,
        index=0,
    )

    def format_gemini(x: str) -> str:
        return x.replace("gemini-2.5-", "").replace("-preview-06-17", " Lite").title()

    def format_openai(x: str) -> str:
        return x.upper()

    def format_mistral(x: str) -> str:
        return x.replace("mistral-", "").replace("-latest", "").title()

    def format_anthropic(x: str) -> str:
        return (
            x.replace("claude-", "")
            .replace("-20250219", "")
            .replace("-20250514", "")
            .title()
        )

    def format_huggingface(x: str) -> str:
        # Extract model name from path
        if "/" in x:
            return x.split("/")[-1].replace("-", " ")
        return x

    def format_ollama(x: str) -> str:
        # Format Ollama model names for display
        return x.replace(":", " ").replace("-", " ").title()

    def format_default(x: str) -> str:
        return x

    if selected_provider == Provider.GOOGLE.value:
        model_options = [model.value for model in GoogleModel]
        format_func = format_gemini
    elif selected_provider == Provider.OPENAI.value:
        model_options = [model.value for model in OpenAIModel]
        format_func = format_openai
    elif selected_provider == Provider.MISTRAL.value:
        model_options = [model.value for model in MistralModel]
        format_func = format_mistral
    elif selected_provider == Provider.ANTHROPIC.value:
        model_options = [model.value for model in AnthropicModel]
        format_func = format_anthropic
    elif selected_provider == Provider.HUGGINGFACE.value:
        model_options = [model.value for model in HuggingFaceModel]
        format_func = format_huggingface
    elif selected_provider == Provider.OLLAMA.value:
        model_options = [model.value for model in OllamaModel]
        format_func = format_ollama
    else:
        model_options = ["Not implemented"]
        format_func = format_default

    selected_model = st.selectbox(
        "Model:",
        options=model_options,
        format_func=format_func,
        index=0,
    )

    streaming = st.toggle("Enable Streaming", value=False)

st.markdown(f"*Powered by {selected_provider.title()}*")

# Main interface
prompt = st.text_area(
    "Enter your prompt:",
    value="Why is the sky blue?",
    height=100,
    placeholder="Ask me anything...",
)

if st.button("✨ Generate", type="primary", use_container_width=True):
    # Create client (accepts both string and enum)
    client = create_client(selected_provider, model=selected_model)

    # Use the prompt directly as a string
    ai_prompt = prompt

    # Show prompt details in an expander
    with st.expander("🔍 Prompt Details", expanded=False):
        st.text(ai_prompt)

    if streaming:
        placeholder = st.empty()
        response_chunks: List[AIResponse] = []

        async def stream_response() -> Tuple[str, List[AIResponse]]:
            response_text = ""
            async for chunk in client.stream_generate_content(ai_prompt):
                response_chunks.append(chunk)
                # Only append content if it's not the final usage-only chunk
                if chunk.content:
                    response_text += chunk.content
                    placeholder.markdown(f"**Response:**\n\n{response_text}▌")
            placeholder.markdown(f"**Response:**\n\n{response_text}")
            return response_text, response_chunks

        response_text, chunks = asyncio.run(stream_response())

        # Combine all chunks into a single response for display
        if chunks:
            # Find the last chunk with usage data, or use the last chunk
            final_usage = None
            for chunk in reversed(chunks):
                if chunk.usage:
                    final_usage = chunk.usage
                    break

            # Create a combined response that matches non-streaming format
            combined_response = AIResponse(
                content=response_text,
                usage=final_usage,
                provider=chunks[0].provider,
                metadata={
                    k: v
                    for k, v in chunks[0].metadata.items()
                    if k != "is_stream_chunk"
                },
            )

            with st.expander("📊 AIResponse Details", expanded=False):
                # Convert provider enum to its value for proper JSON display
                response_dict = combined_response.__dict__.copy()
                if response_dict.get("provider"):
                    response_dict["provider"] = response_dict["provider"].value
                # Convert AIUsage dataclass to dict
                if response_dict.get("usage"):
                    response_dict["usage"] = response_dict["usage"].__dict__
                st.json(response_dict)
    else:
        with st.spinner("Generating..."):
            response = asyncio.run(client.generate_content(ai_prompt))
            st.markdown(f"**Response:**\n\n{response.content}")

            # Show AIResponse details in an expander
            with st.expander("📊 AIResponse Details", expanded=False):
                # Convert provider enum to its value for proper JSON display
                response_dict = response.__dict__.copy()
                if response_dict.get("provider"):
                    response_dict["provider"] = response_dict["provider"].value
                # Convert AIUsage dataclass to dict
                if response_dict.get("usage"):
                    response_dict["usage"] = response_dict["usage"].__dict__
                st.json(response_dict)

# Footer
st.markdown("---")
st.caption(f"Built with Streamlit • Powered by {selected_provider.title()}")
