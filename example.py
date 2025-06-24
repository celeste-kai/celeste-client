import streamlit as st
import asyncio
from celeste_client import create_client
from celeste_client.core.enums import (
    Provider,
    GeminiModel,
    OpenAIModel,
    MistralModel,
    AnthropicModel,
)

st.set_page_config(page_title="Celeste AI", page_icon="🌟", layout="wide")

st.title("🌟 Celeste AI Client")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    def format_provider(x):
        return x.title()

    selected_provider = st.selectbox(
        "Provider:",
        options=[provider.value for provider in Provider],
        format_func=format_provider,
        index=0,
    )

    def format_gemini(x):
        return x.replace("gemini-2.5-", "").replace("-preview-06-17", " Lite").title()

    def format_openai(x):
        return x.upper()

    def format_mistral(x):
        return x.replace("mistral-", "").replace("-latest", "").title()

    def format_anthropic(x):
        return (
            x.replace("claude-", "")
            .replace("-20250219", "")
            .replace("-20250514", "")
            .title()
        )

    def format_default(x):
        return x

    if selected_provider == Provider.GEMINI.value:
        model_options = [model.value for model in GeminiModel]
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
    client = create_client(selected_provider, model=selected_model)

    if streaming:
        placeholder = st.empty()

        async def stream_response():
            response = ""
            async for chunk in client.stream_generate_content(prompt):
                response += chunk
                placeholder.markdown(f"**Response:**\n\n{response}▌")
            placeholder.markdown(f"**Response:**\n\n{response}")

        asyncio.run(stream_response())
    else:
        with st.spinner("Generating..."):
            response = asyncio.run(client.generate_content(prompt))
            st.markdown(f"**Response:**\n\n{response}")

# Footer
st.markdown("---")
st.caption(f"Built with Streamlit • Powered by {selected_provider.title()}")
