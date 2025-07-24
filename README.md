# SRE + DevOps Utility Kit

A Streamlit application with a collection of tools to make your life as an SRE or DevOps engineer easier.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sretoolkit.streamlit.app/)

## Features

This application includes the following tools:

### 1. JSON/XML Formatter & Validator

A powerful tool for working with JSON and XML data.

- **Format & Validate**: Beautify and validate your JSON or XML data.
- **Convert**: Convert data between JSON and XML formats.
- **Multiple Inputs**: Load data from manual input, file upload, or a URL.
- **Tree View**: Visualize your data in a tree-like structure.
- **Sample Data**: Load sample JSON or XML data to get started quickly.

### 2. SRE Quick Utilities Hub

A comprehensive toolkit with a wide range of utilities for SREs and DevOps engineers.

- **Network Tools**: CIDR calculator and IP address validator.
- **Security Tools**: Base64 encoder/decoder and JWT decoder.
- **Time & Date**: Timestamp converter and current time information.
- **Text Tools**: Text utilities and a JSON formatter.
- **Hash & Crypto**: Hash generator and HMAC generator.
- **URL Tools**: URL encoder/decoder and URL parser.
- **System Tools**: Port scanner and system information.
- **Generators**: UUID generator, random password generator, and API key generator.
- **Database Tools**: SQL query formatter and connection string parser.
- **Code Tools**: Regex tester and color code converter.
- **DevOps Tools**: YAML validator, environment variable generator, Docker command generator, and Kubernetes resource generator.
- **Data Tools**: CSV data analyzer, log parser, and data format converter.

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### 3. AI Content Analyzer

A simple interface to interact with a local Ollama API.

- **System Prompt**: Set the context for the AI.
- **Content Analysis**: Analyze, summarize, or ask questions about your content.
