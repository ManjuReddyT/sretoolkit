import streamlit as st
import requests
import json

def ai_analyzer():
    st.subheader("🤖 AI Content Analyzer")

    # Get Ollama settings from session state, with defaults
    ollama_api_url = st.session_state.get("config", {}).get("ollama_api_url", "http://localhost:11434/api/generate")
    ollama_model_name = st.session_state.get("config", {}).get("ollama_model_name", "deepseek-r1:8b")

    # Instructions and input fields
    st.markdown("Enter text below to analyze, summarize, or ask questions about your content.")

    # System prompt for context
    system_prompt = st.text_area(
        "**System Prompt (Optional)**",
        "You are an expert technical analyst. Please analyze the following content.",
        help="Set the context for the AI (e.g., 'You are a cybersecurity expert')."
    )

    # Main content input
    user_content = st.text_area("Enter your content here", height=250)

    if st.button("Analyze Content"):
        if not user_content.strip():
            st.warning("Please enter some content to analyze.")
        else:
            full_prompt = f"{system_prompt}\n\n---\n\n{user_content}"

            headers = {'Content-Type': 'application/json'}
            data = {
                "model": ollama_model_name,
                "prompt": full_prompt,
                "stream": True,
                "options": {"num_predict": 2048}
            }

            try:
                with st.spinner("AI is analyzing..."):
                    response = requests.post(ollama_api_url, headers=headers, json=data, stream=True)
                    response.raise_for_status()

                    st.subheader("Analysis Results")
                    ai_response_placeholder = st.empty()
                    full_ai_response = ""

                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            try:
                                json_data = json.loads(chunk.decode('utf-8'))
                                full_ai_response += json_data.get("response", "")
                                ai_response_placeholder.markdown(full_ai_response)
                            except json.JSONDecodeError:
                                # Handle cases where a chunk is not a valid JSON object
                                pass

                st.success("Analysis complete!")

            except requests.exceptions.ConnectionError:
                st.error("Connection to Ollama API failed. Please ensure Ollama is running and the API URL is correct.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
