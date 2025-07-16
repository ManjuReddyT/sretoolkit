import streamlit as st

st.set_page_config(
    page_title="SRE + DevOps Utility Kit",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.buymeacoffee.com/sretoolkit',
        'Report a bug': "https://github.com/sre-toolkit/sre-toolkit.streamlit.app/issues",
        'About': """
         ### SRE + DevOps Utility Kit
         A collection of tools to make your life as an SRE or DevOps engineer easier.
         Select a tool from the sidebar to begin.
         - **JSON/XML Beautifier**: Format, validate, and convert between JSON and XML.
         - **Quick Utilities**: A hub of tools for networking, security, text manipulation, and more.
         """
    }
)

st.markdown(\"""
<meta name="description" content="A collection of tools for SRE and DevOps engineers, including JSON/XML beautifier, networking utilities, and more.">
<meta name="keywords" content="SRE, DevOps, JSON, XML, beautifier, utilities, networking, security, text manipulation">
<meta name="author" content="SRE Toolkit">
\""", unsafe_allow_html=True)

st.title("🚀 Welcome to SRE + DevOps Utility Kit")
st.write("""
A collection of tools to make your life as an SRE or DevOps engineer easier.

Select a tool from the sidebar to begin.
- **JSON/XML Beautifier**: Format, validate, and convert between JSON and XML.
- **Quick Utilities**: A hub of tools for networking, security, text manipulation, and more.
""")
