import streamlit as st
import pandas as pd
import os
import sys
import json
import tempfile
from datetime import datetime, time
import importlib.util

# Constants
UPLOAD_DIR = "uploads"
REPORTS_DIR = "reports"
CONFIG_FILE = "config.json"
SCRIPTS_DIR = "scripts"

def load_script(script_path):
    """Dynamically loads a Python script module."""
    try:
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        st.error(f"Failed to load script {script_path}: {str(e)}")
        return None

def load_config():
    """Loads the configuration from config.json."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Configuration file {CONFIG_FILE} not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {CONFIG_FILE}")
        return None

def render_input_field(input_config, key_prefix):
    """Renders a single input field based on its configuration."""
    input_type = input_config.get('type', 'text')
    name = input_config['name']
    display_name = input_config.get('display_name', name)
    help_text = input_config.get('help', '')
    required = input_config.get('required', False)
    
    if input_type == 'file':
        accepted_types = input_config.get('accepted_types', None)
        return st.file_uploader(
            f"{display_name}{'*' if required else ''}",
            type=accepted_types,
            help=help_text,
            key=f"{key_prefix}_{name}"
        )
    elif input_type == 'date':
        return st.date_input(
            f"{display_name}{'*' if required else ''}",
            value=None,
            help=help_text,
            key=f"{key_prefix}_{name}"
        )
    elif input_type == 'time':
        default_time = datetime.strptime(
            input_config.get('default', '00:00:00'),
            "%H:%M:%S"
        ).time()
        return st.time_input(
            f"{display_name}{'*' if required else ''}",
            value=default_time,
            help=help_text,
            key=f"{key_prefix}_{name}"
        )
    elif input_type == 'text':
        return st.text_input(
            f"{display_name}{'*' if required else ''}",
            help=help_text,
            key=f"{key_prefix}_{name}"
        )
    # Add more input types as needed

def run_script(script_config, inputs):
    """Runs a script with the provided inputs and handles its output."""
    script_path = script_config['script_path']
    module = load_script(script_path)
    if not module:
        return

    try:
        # Process inputs according to script requirements
        processed_inputs = {}
        
        # Handle file inputs
        for input_config in script_config['inputs']:
            name = input_config['name']
            if input_config['type'] == 'file' and inputs.get(name):
                with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
                    tmp_file.write(inputs[name].getvalue())
                    processed_inputs[name] = tmp_file.name
            else:
                processed_inputs[name] = inputs.get(name)

        # Special handling for date/time inputs
        if 'start_date' in inputs and 'start_time' in inputs:
            if inputs['start_date']:
                processed_inputs['start_datetime'] = datetime.combine(
                    inputs['start_date'], 
                    inputs['start_time']
                )
        if 'end_date' in inputs and 'end_time' in inputs:
            if inputs['end_date']:
                processed_inputs['end_datetime'] = datetime.combine(
                    inputs['end_date'],
                    inputs['end_time']
                )

        # Run the script
        if hasattr(module, 'process_log_lines'):  # For MongoDB parser
            with open(processed_inputs['log_file'], 'r') as file:
                lines = file.readlines()
            results = module.process_log_lines(
                lines,
                processed_inputs.get('start_datetime'),
                processed_inputs.get('end_datetime')
            )
            
            # Create Excel report
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                output_file = tmp_excel.name
                module.write_output_file(output_file, *results)
                module.generate_summary_txt(
                    output_file, len(lines), results[-1],
                    *results[:-1]
                )
                
                # Create download buttons
                with open(output_file, 'rb') as excel_file:
                    excel_data = excel_file.read()
                with open(output_file.replace('.xlsx', '_summary.txt'), 'rb') as summary_file:
                    summary_data = summary_file.read()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Excel Report",
                        excel_data,
                        "analysis_report.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        "Download Summary Report",
                        summary_data,
                        "analysis_summary.txt",
                        "text/plain"
                    )

                # Cleanup
                os.unlink(output_file)
                os.unlink(output_file.replace('.xlsx', '_summary.txt'))

        elif hasattr(module, 'main'):  # Generic script execution
            module.main(processed_inputs)

    except Exception as e:
        st.error(f"Error running script: {str(e)}")
    finally:
        # Cleanup temporary files
        for input_config in script_config['inputs']:
            if input_config['type'] == 'file' and inputs.get(input_config['name']):
                try:
                    os.unlink(processed_inputs[input_config['name']])
                except:
                    pass

def main():
    st.title("SRE Tools")
    
    config = load_config()
    if not config:
        return

    # Script selection
    script_names = {s['name']: s['display_name'] for s in config['scripts']}
    selected_script = st.sidebar.selectbox(
        "Select Tool",
        options=list(script_names.keys()),
        format_func=lambda x: script_names[x]
    )

    # Get selected script configuration
    script_config = next(s for s in config['scripts'] if s['name'] == selected_script)

    # Display script information
    st.header(script_config['display_name'])
    st.write(script_config['description'])
    
    with st.expander("Help"):
        st.markdown(script_config.get('help_text', 'No help available for this tool.'))

    # Render inputs
    st.subheader("Input Parameters")
    inputs = {}
    for input_config in script_config['inputs']:
        inputs[input_config['name']] = render_input_field(input_config, selected_script)

    # Run script button
    if st.button("Run Analysis"):
        required_inputs = [inp['name'] for inp in script_config['inputs'] if inp.get('required', False)]
        missing_inputs = [name for name in required_inputs if not inputs.get(name)]
        
        if missing_inputs:
            st.error(f"Please provide the following required inputs: {', '.join(missing_inputs)}")
        else:
            with st.spinner("Processing..."):
                run_script(script_config, inputs)

if __name__ == "__main__":
    main()
