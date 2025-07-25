import streamlit as st
from page_config import NAV_STRUCTURE

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
         """
    }
)

pg = st.navigation(NAV_STRUCTURE)
pg.run()
