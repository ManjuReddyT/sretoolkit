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
         A comprehensive toolkit for SRE and DevOps engineers.
         
         Available Tools:
         Scripts:
         - **MongoDB Log Parser**: Analyze MongoDB logs and get performance insights
         - **MySQL Log Parser**: Process MySQL slow query logs
         - **Strace Analyzer**: System call analysis and performance diagnostics
         
         Data Processing:
         - **JSON/XML Beautifier**: Format and convert between JSON and XML
         - **Quick Utilities**: Network, security, and text manipulation tools
         """
    }
)

st.markdown("""
<meta name="description" content="A collection of tools for SRE and DevOps engineers, including log parsers, data analyzers, and utility tools.">
<meta name="keywords" content="SRE, DevOps, MongoDB, MySQL, Strace, Log Analysis, JSON, XML, beautifier, utilities, networking, security">
<meta name="author" content="SRE Toolkit">
""", unsafe_allow_html=True)

st.title("🚀 Welcome to SRE + DevOps Utility Kit")
st.write("""
A comprehensive toolkit for SRE and DevOps engineers, featuring:

### Log Analysis Tools
Access these tools in the Scripts section:
- **MongoDB Log Parser**: Analyze MongoDB logs, identify slow queries, and get performance insights
- **MySQL Log Parser**: Process MySQL slow query logs and analyze query performance
- **Strace Analyzer**: Analyze system call patterns and performance bottlenecks

### Data Processing
- **JSON/XML Beautifier**: Format, validate, and convert between JSON and XML
- **Quick Utilities**: A hub for networking, security, and text manipulation tools

### Getting Started
1. Select a tool category from the sidebar
2. Choose your specific tool
3. Follow the intuitive interface to analyze your data

For log analysis tools, visit the Scripts section where you can:
- Upload log files
- Set analysis parameters
- Get detailed reports and insights
- Download results in Excel or text format
""")
