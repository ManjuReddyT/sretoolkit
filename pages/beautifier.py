import streamlit as st
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import collections
import requests
from io import StringIO

# Set page config
st.set_page_config(
    page_title="JSON/XML Formatter & Validator",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for buttons
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    margin: 2px;
}

.data-type-button {
    background-color: #f0f2f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 2px;
    cursor: pointer;
    transition: all 0.2s;
}

.data-type-button:hover {
    background-color: #e5e7eb;
}

.data-type-button.active {
    background-color: #3b82f6;
    color: white;
    border-color: #3b82f6;
}

.operation-button {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 2px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}

.operation-button:hover {
    background-color: #f1f5f9;
}

.operation-button.active {
    background-color: #10b981;
    color: white;
    border-color: #10b981;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_type' not in st.session_state:
    st.session_state.data_type = 'JSON'
if 'operation' not in st.session_state:
    st.session_state.operation = 'Format & Validate'
if 'input_data' not in st.session_state:
    st.session_state.input_data = ''

# Title and description
st.title("🔧 JSON/XML Formatter & Validator")
st.markdown("**The world's leading expert in structured data formatting, validation, and visualization.**")
st.markdown(
    "Instantly beautify, validate, and transform raw JSON and XML data into human-readable, well-indented structures.")


# Utility functions
def format_json(data):
    """Format and validate JSON data"""
    try:
        parsed_json = json.loads(data)
        formatted = json.dumps(parsed_json, indent=2)
        return formatted, None
    except json.JSONDecodeError as e:
        return None, f"JSON Error: {str(e)}"


def format_xml(data):
    """Format and validate XML data"""
    try:
        root = ET.fromstring(data)
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        formatted = reparsed.toprettyxml(indent="  ")
        # Remove empty lines
        formatted = '\n'.join([line for line in formatted.split('\n') if line.strip()])
        return formatted, None
    except ET.ParseError as e:
        return None, f"XML Error: {str(e)}"


def json_to_xml(json_data):
    """Convert JSON to XML"""
    try:
        parsed_json = json.loads(json_data)
        root = ET.Element("root")
        build_xml_from_json(root, parsed_json)
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        formatted = reparsed.toprettyxml(indent="  ")
        # Remove empty lines
        formatted = '\n'.join([line for line in formatted.split('\n') if line.strip()])
        return formatted, None
    except json.JSONDecodeError as e:
        return None, f"JSON Error: {str(e)}"
    except Exception as e:
        return None, f"Conversion Error: {str(e)}"


def xml_to_json(xml_data):
    """Convert XML to JSON"""
    try:
        root = ET.fromstring(xml_data)
        json_data = etree_to_dict(root)
        formatted = json.dumps(json_data, indent=2)
        return formatted, None
    except ET.ParseError as e:
        return None, f"XML Error: {str(e)}"
    except Exception as e:
        return None, f"Conversion Error: {str(e)}"


def build_xml_from_json(parent, json_obj):
    """Helper function to build XML from JSON"""
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            # Clean key name for XML
            clean_key = str(key).replace(' ', '_').replace('-', '_')
            element = ET.SubElement(parent, clean_key)
            build_xml_from_json(element, value)
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            element = ET.SubElement(parent, "item")
            build_xml_from_json(element, item)
    else:
        parent.text = str(json_obj)


def etree_to_dict(t):
    """Helper function to convert XML to dictionary"""
    d = {t.tag: {}}
    children = list(t)
    if children:
        dd = collections.defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {key: value[0] if len(value) == 1 else value for key, value in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def create_tree_view(data, indent=0):
    """Create a tree-like view of the data structure"""
    tree_lines = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                tree_lines.append(f"{prefix}📁 {key}")
                tree_lines.extend(create_tree_view(value, indent + 1))
            else:
                tree_lines.append(f"{prefix}📄 {key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                tree_lines.append(f"{prefix}📁 [{i}]")
                tree_lines.extend(create_tree_view(item, indent + 1))
            else:
                tree_lines.append(f"{prefix}📄 [{i}]: {item}")
    else:
        tree_lines.append(f"{prefix}📄 {data}")

    return tree_lines


def load_from_url(url):
    """Load data from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text, None
    except requests.exceptions.RequestException as e:
        return None, f"URL Error: {str(e)}"


# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# Data Type Selection with buttons
st.sidebar.subheader("📊 Data Type")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📄 JSON", key="json_btn", use_container_width=True):
        st.session_state.data_type = 'JSON'

with col2:
    if st.button("🏷️ XML", key="xml_btn", use_container_width=True):
        st.session_state.data_type = 'XML'

st.sidebar.write(f"**Selected:** {st.session_state.data_type}")

# Operation Selection with buttons
st.sidebar.subheader("⚙️ Operations")

if st.sidebar.button("✨ Format & Validate", key="format_btn", use_container_width=True):
    st.session_state.operation = 'Format & Validate'

if st.sidebar.button("🔄 JSON → XML", key="j2x_btn", use_container_width=True):
    st.session_state.operation = 'Convert JSON to XML'

if st.sidebar.button("🔄 XML → JSON", key="x2j_btn", use_container_width=True):
    st.session_state.operation = 'Convert XML to JSON'

st.sidebar.write(f"**Selected:** {st.session_state.operation}")

# Tree view toggle
show_tree_view = st.sidebar.checkbox("🌳 Show Tree View", value=False)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Input")

    # Input method tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✏️ Manual", "📁 File", "🌐 URL", "📋 Samples"])

    with tab1:
        # Manual input text area
        input_data = st.text_area(
            "Enter your JSON or XML data:",
            height=300,
            value=st.session_state.input_data,
            key='manual_input'
        )
        if input_data != st.session_state.input_data:
            st.session_state.input_data = input_data

    with tab2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a JSON or XML file",
            type=['json', 'xml', 'txt'],
            help="Upload a file containing JSON or XML data"
        )

        if uploaded_file is not None:
            try:
                # Read file content
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                file_content = stringio.read()
                st.session_state.input_data = file_content
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                st.text_area("File content preview:",
                             value=file_content[:500] + "..." if len(file_content) > 500 else file_content, height=200,
                             disabled=True)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    with tab3:
        # URL input
        url_input = st.text_input(
            "Enter URL to load data from:",
            placeholder="https://example.com/data.json",
            help="Enter a URL that returns JSON or XML data"
        )

        if st.button("🔗 Load from URL", disabled=not url_input):
            if url_input:
                with st.spinner("Loading data from URL..."):
                    url_data, url_error = load_from_url(url_input)
                    if url_error:
                        st.error(f"❌ {url_error}")
                    else:
                        st.session_state.input_data = url_data
                        st.success("✅ Data loaded from URL successfully!")
                        st.text_area("URL content preview:",
                                     value=url_data[:500] + "..." if len(url_data) > 500 else url_data, height=200,
                                     disabled=True)

    with tab4:
        # Sample data buttons
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            if st.button("📄 Load Sample JSON", use_container_width=True):
                sample_json = '{"person":{"name":"Alice","age":30,"contacts":[{"type":"email","value":"alice@example.com"},{"type":"phone","value":"123-456"}]}}'
                st.session_state.input_data = sample_json
                st.success("✅ Sample JSON loaded!")

        with col_s2:
            if st.button("🏷️ Load Sample XML", use_container_width=True):
                sample_xml = '<person><name>Alice</name><age>30</age><contacts><contact><type>email</type><value>alice@example.com</value></contact><contact><type>phone</type><value>123-456</value></contact></contacts></person>'
                st.session_state.input_data = sample_xml
                st.success("✅ Sample XML loaded!")

with col2:
    st.header("📤 Output")

    if st.session_state.input_data.strip():
        if st.session_state.operation == "Format & Validate":
            if st.session_state.data_type == "JSON":
                result, error = format_json(st.session_state.input_data)
            else:  # XML
                result, error = format_xml(st.session_state.input_data)
        elif st.session_state.operation == "Convert JSON to XML":
            result, error = json_to_xml(st.session_state.input_data)
        elif st.session_state.operation == "Convert XML to JSON":
            result, error = xml_to_json(st.session_state.input_data)

        if error:
            st.error(f"❌ {error}")

            # Provide suggestions for common errors
            st.subheader("💡 Suggestions:")
            if "JSON Error" in error:
                st.write("- Check for missing or extra commas")
                st.write("- Ensure all strings are properly quoted")
                st.write("- Verify bracket/brace matching: {}, []")
                st.write("- Remove trailing commas")
            elif "XML Error" in error:
                st.write("- Check for unclosed tags")
                st.write("- Ensure proper tag nesting")
                st.write("- Verify attribute syntax")
                st.write("- Check for special characters that need escaping")
        else:
            st.success("✅ Data is valid and formatted!")

            # Display formatted result
            st.code(result,
                    language='json' if 'json' in st.session_state.operation.lower() or st.session_state.data_type == 'JSON' else 'xml')

            # Show tree view if requested
            if show_tree_view:
                st.subheader("🌳 Tree View")
                try:
                    if st.session_state.operation == "Convert XML to JSON" or (
                            st.session_state.operation == "Format & Validate" and st.session_state.data_type == "JSON"):
                        parsed_data = json.loads(result)
                    else:
                        # For XML, parse it to create a tree view
                        root = ET.fromstring(result)
                        parsed_data = etree_to_dict(root)

                    tree_lines = create_tree_view(parsed_data)
                    tree_text = '\n'.join(tree_lines)
                    st.text(tree_text)
                except Exception as e:
                    st.warning(f"Could not generate tree view: {e}")

            # Download button
            file_extension = 'json' if 'json' in st.session_state.operation.lower() or st.session_state.data_type == 'JSON' else 'xml'
            st.download_button(
                label=f"📥 Download {file_extension.upper()}",
                data=result,
                file_name=f"formatted_data.{file_extension}",
                mime=f"application/{file_extension}"
            )
    else:
        st.info("👆 Enter some data in the input area to get started!")
