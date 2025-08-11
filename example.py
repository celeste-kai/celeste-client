import asyncio

import streamlit as st
from celeste_client import create_client
from celeste_core import Provider, list_models
from celeste_core.enums.capability import Capability

st.set_page_config(page_title="Celeste AI", page_icon="🌟", layout="wide")
st.title("🌟 Celeste AI Client")
providers = sorted(
    {m.provider for m in list_models(capability=Capability.TEXT_GENERATION)},
    key=lambda p: p.value,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox(
        "Provider:", [p.value for p in providers], format_func=str.title
    )
    models = list_models(
        provider=Provider(provider), capability=Capability.TEXT_GENERATION
    )
    model_names = [m.display_name or m.id for m in models]
    selected_idx = st.selectbox(
        "Model:", range(len(models)), format_func=lambda i: model_names[i]
    )
    model = models[selected_idx].id
    streaming = st.toggle("Enable Streaming")

st.markdown(f"*Powered by {provider.title()}*")
prompt = st.text_area(
    "Enter your prompt:",
    "Why is the sky blue?",
    height=100,
    placeholder="Ask me anything...",
)

if st.button("✨ Generate", type="primary", use_container_width=True):
    client = create_client(Provider(provider), model=model)

    async def generate() -> None:
        content = ""
        if streaming:
            placeholder = st.empty()
            async for chunk in client.stream_generate_content(prompt):
                if chunk.content:
                    content += chunk.content
                    placeholder.markdown(f"**Response:**\n\n{content}▌")
            placeholder.markdown(f"**Response:**\n\n{content}")
        else:
            with st.spinner("Generating..."):
                response = await client.generate_content(prompt)
                st.markdown(f"**Response:**\n\n{response.content}")

    asyncio.run(generate())
st.markdown("---")
st.caption("Built with Streamlit • Powered by Celeste")
