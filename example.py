import streamlit as st
import asyncio
from celeste_client import create_client
from celeste_client.core.enums import GeminiModel

st.set_page_config(page_title="Celeste AI", page_icon="🌟", layout="wide")

st.title("🌟 Celeste AI Client")
st.markdown("*Powered by Google Gemini*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Model:",
        options=[model.value for model in GeminiModel],
        format_func=lambda x: x.replace("gemini-2.5-", "")
        .replace("-preview-06-17", " Lite")
        .title(),
        index=0,
    )

    streaming = st.toggle("Enable Streaming", value=False)

# Main interface
prompt = st.text_area(
    "Enter your prompt:",
    value="Why is the sky blue?",
    height=100,
    placeholder="Ask me anything...",
)

if st.button("✨ Generate", type="primary", use_container_width=True):
    client = create_client("gemini", model=selected_model)

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
st.caption("Built with Streamlit • Powered by Google Gemini")
