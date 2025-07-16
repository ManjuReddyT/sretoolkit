# SRE Toolkit

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sretoolkit.streamlit.app/)

An all-in-one toolkit for Site Reliability Engineers (SREs) and developers to streamline common tasks. This Streamlit application provides a collection of tools to simplify your workflows, starting with a powerful JSON/XML formatter and validator.

## Features

### 🔧 JSON/XML Formatter & Validator

The initial tool in the SRE Toolkit is a robust JSON/XML utility that allows you to:

- **Format & Validate:** Instantly beautify and validate raw JSON and XML data. The tool provides clear error messages to help you quickly identify and fix issues.
- **Convert:** Seamlessly convert data between JSON and XML formats.
- **Multiple Input Methods:**
    - **Manual:** Type or paste your data directly.
    - **File Upload:** Upload JSON or XML files from your local machine.
    - **URL:** Fetch data directly from a URL.
    - **Sample Data:** Load sample data to quickly test the functionality.
- **Tree View:** Visualize your structured data in a hierarchical tree view for better understanding.

## How to run it on your own machine

1. **Install the requirements:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**

   ```bash
   streamlit run streamlit_app.py
   ```

## Contributing

Contributions are welcome! If you have ideas for new tools or improvements, please open an issue or submit a pull request.
