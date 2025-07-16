import streamlit as st

st.set_page_config(
    page_title="SRE Toolkit",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ SRE Toolkit")

st.markdown("""
An all-in-one toolkit for Site Reliability Engineers (SREs) and developers to streamline common tasks.
This Streamlit application provides a collection of tools to simplify your workflows.

### 👈 Select a tool from the sidebar to get started.

Current tools include:
- **JSON/XML Formatter & Validator:** A powerful utility to format, validate, and convert data between JSON and XML.

Have an idea for a new tool? Feel free to [contribute](https://github.com/your-repo/sre-toolkit)!
""")
