import streamlit as st
import base64
import jwt
import datetime
import ipaddress
import json
import hashlib
import hmac
import secrets
import urllib.parse
import re
import uuid
import socket
import time
from pages.ai_analyzer import ai_analyzer

# Configure page
st.set_page_config(page_title="SRE Quick Utilities Hub", page_icon="🔧", layout="wide")

st.title("🔧 SRE Quick Utilities Hub")
st.markdown("*A comprehensive toolkit for Site Reliability Engineers*")

# Create tabs for main navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "🌐 Network Tools", 
    "🔐 Security Tools", 
    "⏰ Time & Date", 
    "🔤 Text Tools", 
    "🔍 Hash & Crypto", 
    "🌍 URL Tools", 
    "📊 System Tools",
    "🎲 Generators",
    "🗄️ Database Tools",
    "📋 Code Tools",
    "🔧 DevOps Tools",
    "📈 Data Tools",
    "🤖 AI Tools"
])

# Network Tools Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CIDR Calculator")
        cidr_input = st.text_input("Enter CIDR (e.g. 192.168.1.0/24)")
        if cidr_input:
            try:
                net = ipaddress.ip_network(cidr_input, strict=False)
                st.success("✅ Valid CIDR")
                st.write("**Network address:**", net.network_address)
                st.write("**Broadcast address:**", net.broadcast_address)
                st.write("**Number of addresses:**", net.num_addresses)
                st.write("**Netmask:**", net.netmask)
                st.write("**Prefix length:**", net.prefixlen)
                
                # Show first and last 5 IPs
                hosts = list(net.hosts())
                if hosts:
                    st.write("**First 5 hosts:**", str(hosts[:5]))
                    if len(hosts) > 5:
                        st.write("**Last 5 hosts:**", str(hosts[-5:]))
            except Exception as e:
                st.error(f"❌ Invalid CIDR: {e}")
    
    with col2:
        st.subheader("IP Address Validator")
        ip_input = st.text_input("Enter IP Address")
        if ip_input:
            try:
                ip = ipaddress.ip_address(ip_input)
                st.success("✅ Valid IP Address")
                st.write("**Type:**", "IPv4" if isinstance(ip, ipaddress.IPv4Address) else "IPv6")
                st.write("**Is private:**", ip.is_private)
                st.write("**Is loopback:**", ip.is_loopback)
                st.write("**Is multicast:**", ip.is_multicast)
                
                # DNS lookup
                try:
                    hostname = socket.gethostbyaddr(str(ip))[0]
                    st.write("**Hostname:**", hostname)
                except:
                    st.write("**Hostname:**", "No reverse DNS found")
            except Exception as e:
                st.error(f"❌ Invalid IP: {e}")

# Security Tools Tab
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Base64 Encode/Decode")
        b64_mode = st.radio("Mode", ["Encode", "Decode"], key="b64_mode")
        b64_text = st.text_area("Input Text", key="b64_text")
        if b64_text:
            try:
                if b64_mode == "Encode":
                    encoded = base64.b64encode(b64_text.encode()).decode()
                    st.code(encoded)
                    st.caption(f"Length: {len(encoded)} characters")
                else:
                    decoded = base64.b64decode(b64_text.encode()).decode()
                    st.code(decoded)
                    st.caption(f"Length: {len(decoded)} characters")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col2:
        st.subheader("JWT Decoder")
        token = st.text_area("Enter JWT Token", key="jwt_token")
        if token:
            try:
                header = jwt.get_unverified_header(token)
                payload = jwt.decode(token, options={"verify_signature": False})
                
                col_header, col_payload = st.columns(2)
                with col_header:
                    st.write("**Header:**")
                    st.json(header)
                with col_payload:
                    st.write("**Payload:**")
                    st.json(payload)
                
                # Check expiration
                if 'exp' in payload:
                    exp_time = datetime.datetime.fromtimestamp(payload['exp'])
                    now = datetime.datetime.now()
                    if exp_time < now:
                        st.error("⚠️ Token is expired")
                    else:
                        st.success(f"✅ Token expires at: {exp_time}")
            except Exception as e:
                st.error(f"❌ Invalid JWT: {e}")

# Time & Date Tab
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Timestamp Converter")
        time_mode = st.radio("Mode", ["Epoch to Human-readable", "Human-readable to Epoch"], key="time_mode")
        
        if time_mode == "Epoch to Human-readable":
            epoch = st.number_input("Enter Epoch (seconds)", step=1.0, key="epoch_input")
            if epoch:
                dt = datetime.datetime.fromtimestamp(epoch)
                dt_utc = datetime.datetime.utcfromtimestamp(epoch)
                st.write("**Local time:**", dt.strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**UTC time:**", dt_utc.strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**ISO format:**", dt.isoformat())
        else:
            date_str = st.text_input("Enter date (YYYY-MM-DD HH:MM:SS)", key="date_input")
            if date_str:
                try:
                    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    epoch = int(dt.timestamp())
                    st.write("**Epoch (seconds):**", epoch)
                    st.write("**Epoch (milliseconds):**", epoch * 1000)
                except Exception as e:
                    st.error(f"❌ Invalid date format: {e}")
    
    with col2:
        st.subheader("Current Time Info")
        if st.button("Get Current Time", key="current_time"):
            now = datetime.datetime.now()
            utc_now = datetime.datetime.utcnow()
            
            st.write("**Current Local Time:**", now.strftime("%Y-%m-%d %H:%M:%S"))
            st.write("**Current UTC Time:**", utc_now.strftime("%Y-%m-%d %H:%M:%S"))
            st.write("**Current Epoch:**", int(now.timestamp()))
            st.write("**Current Epoch (ms):**", int(now.timestamp() * 1000))
            st.write("**ISO Format:**", now.isoformat())

# Text Tools Tab
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Utilities")
        text_input = st.text_area("Enter text for analysis", key="text_analysis")
        if text_input:
            st.write("**Character count:**", len(text_input))
            st.write("**Word count:**", len(text_input.split()))
            st.write("**Line count:**", len(text_input.splitlines()))
            st.write("**Uppercase:**", text_input.upper())
            st.write("**Lowercase:**", text_input.lower())
            st.write("**Title case:**", text_input.title())
    
    with col2:
        st.subheader("JSON Formatter")
        json_input = st.text_area("Enter JSON to format", key="json_input")
        if json_input:
            try:
                parsed = json.loads(json_input)
                formatted = json.dumps(parsed, indent=2)
                st.code(formatted, language="json")
                st.caption("✅ Valid JSON")
            except Exception as e:
                st.error(f"❌ Invalid JSON: {e}")

# Hash & Crypto Tab
with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hash Generator")
        hash_input = st.text_area("Enter text to hash", key="hash_input")
        if hash_input:
            text_bytes = hash_input.encode('utf-8')
            st.write("**MD5:**", hashlib.md5(text_bytes).hexdigest())
            st.write("**SHA1:**", hashlib.sha1(text_bytes).hexdigest())
            st.write("**SHA256:**", hashlib.sha256(text_bytes).hexdigest())
            st.write("**SHA512:**", hashlib.sha512(text_bytes).hexdigest())
    
    with col2:
        st.subheader("HMAC Generator")
        hmac_text = st.text_input("Text to sign", key="hmac_text")
        hmac_key = st.text_input("Secret key", type="password", key="hmac_key")
        if hmac_text and hmac_key:
            hmac_sha256 = hmac.new(hmac_key.encode(), hmac_text.encode(), hashlib.sha256).hexdigest()
            st.write("**HMAC-SHA256:**", hmac_sha256)

# URL Tools Tab
with tab6:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("URL Encode/Decode")
        url_mode = st.radio("Mode", ["Encode", "Decode"], key="url_mode")
        url_input = st.text_area("Enter URL or text", key="url_input")
        if url_input:
            try:
                if url_mode == "Encode":
                    encoded = urllib.parse.quote(url_input)
                    st.code(encoded)
                else:
                    decoded = urllib.parse.unquote(url_input)
                    st.code(decoded)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col2:
        st.subheader("URL Parser")
        url_parse_input = st.text_input("Enter URL to parse", key="url_parse")
        if url_parse_input:
            try:
                parsed = urllib.parse.urlparse(url_parse_input)
                st.write("**Scheme:**", parsed.scheme)
                st.write("**Netloc:**", parsed.netloc)
                st.write("**Path:**", parsed.path)
                st.write("**Query:**", parsed.query)
                st.write("**Fragment:**", parsed.fragment)
                
                if parsed.query:
                    params = urllib.parse.parse_qs(parsed.query)
                    st.write("**Query Parameters:**")
                    for key, value in params.items():
                        st.write(f"  - {key}: {value}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# System Tools Tab
with tab7:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Port Scanner")
        host = st.text_input("Host to scan", value="127.0.0.1", key="port_host")
        ports = st.text_input("Ports (comma-separated)", value="22,80,443", key="port_list")
        
        if st.button("Scan Ports", key="scan_ports"):
            if host and ports:
                port_list = [int(p.strip()) for p in ports.split(',')]
                for port in port_list:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        st.success(f"✅ Port {port}: Open")
                    else:
                        st.error(f"❌ Port {port}: Closed")
                    sock.close()
    
    with col2:
        st.subheader("System Information")
        if st.button("Get System Info", key="system_info"):
            import platform
            import os
            
            st.write("**Platform:**", platform.platform())
            st.write("**Python Version:**", platform.python_version())
            st.write("**Current Directory:**", os.getcwd())
            st.write("**Environment Variables:**")
            for key in sorted(os.environ.keys())[:10]:  # Show first 10 env vars
                st.write(f"  - {key}: {os.environ[key][:50]}...")

# Generators Tab
with tab8:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("UUID Generator")
        if st.button("Generate UUID4", key="gen_uuid4"):
            st.code(str(uuid.uuid4()))
        
        if st.button("Generate UUID1", key="gen_uuid1"):
            st.code(str(uuid.uuid1()))
        
        st.subheader("Random Password")
        password_length = st.slider("Password Length", 8, 64, 16, key="pwd_length")
        if st.button("Generate Password", key="gen_password"):
            import string
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            password = ''.join(secrets.choice(alphabet) for _ in range(password_length))
            st.code(password)
    
    with col2:
        st.subheader("Random Data")
        if st.button("Generate Random Hex (32 bytes)", key="gen_hex"):
            st.code(secrets.token_hex(32))
        
        if st.button("Generate Random Base64 (32 bytes)", key="gen_b64"):
            st.code(secrets.token_urlsafe(32))
        
        st.subheader("API Key Generator")
        key_length = st.slider("Key Length", 16, 64, 32, key="key_length")
        if st.button("Generate API Key", key="gen_api_key"):
            api_key = secrets.token_urlsafe(key_length)
            st.code(api_key)

# Database Tools Tab
with tab9:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SQL Query Formatter")
        sql_input = st.text_area("Enter SQL query to format", key="sql_input", height=150)
        if sql_input:
            # Basic SQL formatting
            formatted_sql = sql_input.replace(',', ',\n  ').replace('SELECT', 'SELECT\n  ')
            formatted_sql = formatted_sql.replace('FROM', '\nFROM\n  ').replace('WHERE', '\nWHERE\n  ')
            formatted_sql = formatted_sql.replace('ORDER BY', '\nORDER BY\n  ')
            formatted_sql = formatted_sql.replace('GROUP BY', '\nGROUP BY\n  ')
            formatted_sql = formatted_sql.replace('HAVING', '\nHAVING\n  ')
            formatted_sql = formatted_sql.replace('JOIN', '\nJOIN\n  ')
            st.code(formatted_sql, language="sql")
    
    with col2:
        st.subheader("Connection String Parser")
        conn_str = st.text_input("Enter connection string", key="conn_str")
        if conn_str:
            # Parse common connection string formats
            if "://" in conn_str:
                # URL format like postgresql://user:pass@host:port/db
                parts = conn_str.split("://")
                if len(parts) == 2:
                    protocol = parts[0]
                    remainder = parts[1]
                    
                    st.write("**Protocol:**", protocol)
                    
                    if "@" in remainder:
                        auth_part, host_part = remainder.split("@", 1)
                        if ":" in auth_part:
                            user, password = auth_part.split(":", 1)
                            st.write("**User:**", user)
                            st.write("**Password:**", "***" if password else "None")
                        
                        if "/" in host_part:
                            host_port, database = host_part.split("/", 1)
                            st.write("**Database:**", database)
                        else:
                            host_port = host_part
                        
                        if ":" in host_port:
                            host, port = host_port.split(":", 1)
                            st.write("**Host:**", host)
                            st.write("**Port:**", port)
                        else:
                            st.write("**Host:**", host_port)

# Code Tools Tab
with tab10:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regex Tester")
        regex_pattern = st.text_input("Enter regex pattern", key="regex_pattern")
        regex_text = st.text_area("Enter test text", key="regex_text", height=100)
        regex_flags = st.multiselect("Flags", ["IGNORECASE", "MULTILINE", "DOTALL"], key="regex_flags")
        
        if regex_pattern and regex_text:
            try:
                flags = 0
                if "IGNORECASE" in regex_flags:
                    flags |= re.IGNORECASE
                if "MULTILINE" in regex_flags:
                    flags |= re.MULTILINE
                if "DOTALL" in regex_flags:
                    flags |= re.DOTALL
                
                matches = re.findall(regex_pattern, regex_text, flags)
                st.write(f"**Matches found:** {len(matches)}")
                if matches:
                    for i, match in enumerate(matches[:10]):  # Show first 10 matches
                        st.write(f"  {i+1}. `{match}`")
            except Exception as e:
                st.error(f"❌ Regex error: {e}")
    
    with col2:
        st.subheader("Color Code Converter")
        color_input = st.text_input("Enter color code (#hex, rgb, hsl)", key="color_input")
        if color_input:
            # Basic hex color validation and conversion
            if color_input.startswith('#') and len(color_input) == 7:
                hex_code = color_input
                try:
                    # Convert hex to RGB
                    r = int(hex_code[1:3], 16)
                    g = int(hex_code[3:5], 16)
                    b = int(hex_code[5:7], 16)
                    
                    st.write("**HEX:**", hex_code)
                    st.write("**RGB:**", f"rgb({r}, {g}, {b})")
                    st.write("**RGB Values:**", f"R: {r}, G: {g}, B: {b}")
                    
                    # Show color preview
                    st.markdown(f'<div style="width: 100px; height: 50px; background-color: {hex_code}; border: 1px solid #ccc;"></div>', unsafe_allow_html=True)
                except:
                    st.error("❌ Invalid hex color")
            elif color_input.startswith('rgb('):
                st.write("**RGB:**", color_input)
                # Extract RGB values
                try:
                    rgb_values = color_input.replace('rgb(', '').replace(')', '').split(',')
                    r, g, b = [int(x.strip()) for x in rgb_values]
                    hex_code = f"#{r:02x}{g:02x}{b:02x}"
                    st.write("**HEX:**", hex_code)
                    st.write("**RGB Values:**", f"R: {r}, G: {g}, B: {b}")
                except:
                    st.error("❌ Invalid RGB format")

# DevOps Tools Tab
with tab11:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("YAML Validator")
        yaml_input = st.text_area("Enter YAML to validate", key="yaml_input", height=200)
        if yaml_input:
            try:
                import yaml
                parsed = yaml.safe_load(yaml_input)
                st.success("✅ Valid YAML")
                st.json(parsed)
            except Exception as e:
                st.error(f"❌ Invalid YAML: {e}")
    
        st.subheader("Environment Variable Generator")
        env_key = st.text_input("Variable name", key="env_key")
        env_value = st.text_input("Variable value", key="env_value")
        env_format = st.selectbox("Format", ["bash", "docker", "kubernetes", "env file"], key="env_format")
        
        if env_key and env_value:
            if env_format == "bash":
                st.code(f'export {env_key}="{env_value}"')
            elif env_format == "docker":
                st.code(f'ENV {env_key}="{env_value}"')
            elif env_format == "kubernetes":
                st.code(f'- name: {env_key}\n  value: "{env_value}"')
            else:  # env file
                st.code(f'{env_key}={env_value}')
    
    with col2:
        st.subheader("Docker Command Generator")
        docker_image = st.text_input("Docker image", key="docker_image")
        docker_ports = st.text_input("Port mapping (host:container)", key="docker_ports")
        docker_volumes = st.text_input("Volume mapping (host:container)", key="docker_volumes")
        docker_env = st.text_input("Environment variables (KEY=VALUE)", key="docker_env")
        docker_name = st.text_input("Container name", key="docker_name")
        
        if docker_image:
            cmd = f"docker run"
            if docker_name:
                cmd += f" --name {docker_name}"
            if docker_ports:
                cmd += f" -p {docker_ports}"
            if docker_volumes:
                cmd += f" -v {docker_volumes}"
            if docker_env:
                cmd += f" -e {docker_env}"
            cmd += f" {docker_image}"
            st.code(cmd)
        
        st.subheader("Kubernetes Resource Generator")
        k8s_type = st.selectbox("Resource type", ["deployment", "service", "configmap", "secret"], key="k8s_type")
        k8s_name = st.text_input("Resource name", key="k8s_name")
        k8s_namespace = st.text_input("Namespace", value="default", key="k8s_namespace")
        
        if k8s_name:
            if k8s_type == "deployment":
                yaml_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {k8s_name}
  namespace: {k8s_namespace}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {k8s_name}
  template:
    metadata:
      labels:
        app: {k8s_name}
    spec:
      containers:
      - name: {k8s_name}
        image: nginx:latest
        ports:
        - containerPort: 80"""
                st.code(yaml_content, language="yaml")

# Data Tools Tab
with tab12:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CSV Data Analyzer")
        csv_input = st.text_area("Paste CSV data", key="csv_input", height=150)
        if csv_input:
            try:
                import csv
                import io
                
                # Parse CSV
                csv_reader = csv.reader(io.StringIO(csv_input))
                rows = list(csv_reader)
                
                if rows:
                    headers = rows[0]
                    data_rows = rows[1:]
                    
                    st.write("**Headers:**", ", ".join(headers))
                    st.write("**Total rows:**", len(data_rows))
                    st.write("**Total columns:**", len(headers))
                    
                    # Show first few rows
                    st.write("**First 5 rows:**")
                    for i, row in enumerate(data_rows[:5]):
                        st.write(f"  Row {i+1}: {row}")
                    
                    # Column analysis
                    for col_idx, header in enumerate(headers):
                        values = [row[col_idx] for row in data_rows if col_idx < len(row)]
                        non_empty = [v for v in values if v.strip()]
                        st.write(f"**{header}:** {len(non_empty)} non-empty values")
            except Exception as e:
                st.error(f"❌ CSV parsing error: {e}")
    
    with col2:
        st.subheader("Log Parser")
        log_input = st.text_area("Paste log data", key="log_input", height=150)
        log_format = st.selectbox("Log format", ["Apache/Nginx", "JSON", "Custom"], key="log_format")
        
        if log_input:
            lines = log_input.strip().split('\n')
            st.write("**Total lines:**", len(lines))
            
            if log_format == "Apache/Nginx":
                # Basic Apache/Nginx log analysis
                ips = []
                status_codes = []
                for line in lines:
                    # Extract IP (first field)
                    parts = line.split(' ')
                    if parts:
                        ips.append(parts[0])
                    
                    # Extract status code (rough pattern)
                    if '" ' in line:
                        after_quote = line.split('" ')[1]
                        if after_quote:
                            status_codes.append(after_quote.split(' ')[0])
                
                # Count unique IPs
                unique_ips = set(ips)
                st.write("**Unique IPs:**", len(unique_ips))
                
                # Count status codes
                status_counts = {}
                for code in status_codes:
                    status_counts[code] = status_counts.get(code, 0) + 1
                
                st.write("**Status codes:**")
                for code, count in sorted(status_counts.items()):
                    st.write(f"  {code}: {count}")
            
            elif log_format == "JSON":
                # JSON log analysis
                json_logs = []
                for line in lines:
                    try:
                        log_entry = json.loads(line)
                        json_logs.append(log_entry)
                    except:
                        continue
                
                if json_logs:
                    st.write("**Valid JSON logs:**", len(json_logs))
                    
                    # Find common keys
                    all_keys = set()
                    for log in json_logs:
                        all_keys.update(log.keys())
                    
                    st.write("**Common fields:**", ", ".join(sorted(all_keys)))
        
        st.subheader("Data Format Converter")
        input_format = st.selectbox("Input format", ["JSON", "CSV", "YAML"], key="input_format")
        output_format = st.selectbox("Output format", ["JSON", "CSV", "YAML", "XML"], key="output_format")
        convert_data = st.text_area("Data to convert", key="convert_data", height=100)
        
        if convert_data and input_format != output_format:
            try:
                if input_format == "JSON":
                    data = json.loads(convert_data)
                elif input_format == "CSV":
                    csv_reader = csv.DictReader(io.StringIO(convert_data))
                    data = list(csv_reader)
                elif input_format == "YAML":
                    data = yaml.safe_load(convert_data)
                
                if output_format == "JSON":
                    result = json.dumps(data, indent=2)
                    st.code(result, language="json")
                elif output_format == "YAML":
                    result = yaml.dump(data, default_flow_style=False)
                    st.code(result, language="yaml")
                elif output_format == "XML":
                    # Basic XML conversion for simple data
                    st.code(f"<root>{data}</root>", language="xml")
                    st.caption("Note: Basic XML conversion - complex data may need custom formatting")
            except Exception as e:
                st.error(f"❌ Conversion error: {e}")

# AI Tools Tab
with tab13:
    ai_analyzer()

# Footer
st.markdown("---")
st.markdown("*🔧 SRE Quick Utilities Hub - Making SRE tasks easier, one tool at a time*")