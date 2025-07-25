import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import os
from datetime import datetime, timedelta
import tempfile
import time
import plotly.io as pio
import sys
import json
import logging # Import logging
import re # Import regex for error code parsing
import urllib.parse # Needed for URL encoding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Add parent directory to path if pdf_exporter is located there
# Ensure pdf_exporter.py exists in the parent directory or adjust the path accordingly
try:
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # Add the parent directory to sys.path if it's not already there
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Now try importing from pdf_exporter (assuming it's in the parent directory)
    from pdf_exporter import JMeterPDFReport, generate_report, create_error_breakdown_table
    PDF_EXPORTER_AVAILABLE = True
    logging.info("Successfully imported pdf_exporter.")
except ImportError as e:
    logging.warning(f"pdf_exporter.py not found or contains errors. PDF export functionality will be disabled. Error: {e}")
    # Don't use st.warning here as it might run before the main app layout
    PDF_EXPORTER_AVAILABLE = False
    # Define dummy functions if PDF exporter is not available to avoid NameErrors
    class JMeterPDFReport:
        def __init__(self, *args, **kwargs): pass
        def add_page(self): pass
        def chapter_title(self, *args, **kwargs): pass
        def chapter_body(self, *args, **kwargs): pass
        def add_table(self, *args, **kwargs): pass
        def add_plotly_fig(self, *args, **kwargs): pass
        def output(self, *args, **kwargs): pass

    def generate_report(*args, **kwargs):
        logging.error("generate_report called but pdf_exporter is not available.")
        return None
    def create_error_breakdown_table(*args, **kwargs):
        logging.warning("create_error_breakdown_table called but pdf_exporter is not available.")
        return pd.DataFrame()

# Set page configuration
st.set_page_config(
    page_title="Enhanced JMeter Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A; /* Dark Blue */
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1F2937; /* Dark Gray */
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #D1D5DB; /* Light Gray */
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6; /* Lighter Gray */
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
        height: 100%; /* Make cards in a row equal height */
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280; /* Medium Gray */
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1F2937; /* Dark Gray */
        word-wrap: break-word; /* Wrap long values if needed */
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: bold;
    }
    /* Specific styling for Duration/Start/End time */
    .time-info-metric .metric-label { /* Target label within this specific container */
         font-size: 1rem; /* Slightly larger label */
         font-weight: bold;
         color: #4B5563; /* Darker Gray */
    }
     .time-info-metric .metric-value { /* Target value within this specific container */
         font-size: 1.1rem; /* Slightly smaller value */
         font-weight: normal;
         color: #1F2937;
    }

    .success {
        color: #10B981; /* Green */
        font-weight: bold;
    }
    .warning {
        color: #F59E0B; /* Amber */
        font-weight: bold;
    }
    .error {
        color: #EF4444; /* Red */
        font-weight: bold;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #E5E7EB; /* Light Gray */
    }
    .aggregate-table th {
        background-color: #E5E7EB; /* Light Gray */
        font-weight: bold;
        text-align: left;
    }
    .aggregate-table tr:nth-child(even) {
        background-color: #F9FAFB; /* Very Light Gray */
    }
    .comparison-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F2937; /* Dark Gray */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .improved-value {
        color: #10B981; /* Green */
        font-weight: bold;
    }
    .degraded-value {
        color: #EF4444; /* Red */
        font-weight: bold;
    }
    .neutral-value {
        color: #6B7280; /* Medium Gray */
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB; /* Blue */
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8; /* Darker Blue */
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #059669; /* Emerald Green */
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #047857; /* Darker Emerald Green */
    }
    /* Style Streamlit elements */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}
    /* Ensure multiselect takes reasonable space */
    .stMultiSelect [data-baseweb="select"] > div {
        max-height: 150px; /* Limit dropdown height */
        overflow-y: auto;
    }
    /* Adjust filter layout */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        padding-right: 10px; /* Add some space between filter columns */
    }


</style>
""", unsafe_allow_html=True)

# --- Constants ---
# Define standard columns plus the newly confirmed ones
EXPECTED_COLUMNS = [
    'timeStamp', 'elapsed', 'label', 'responseCode', 'success',
    'Latency', 'Connect', 'bytes', 'sentBytes', 'grpThreads', 'allThreads',
    'URL', 'responseMessage', 'failureMessage', 'threadName', 'dataType', 'IdleTime'
]
# Minimum required columns for core functionality
REQUIRED_COLUMNS = ['timeStamp', 'elapsed', 'label', 'responseCode', 'success']


# --- Utility Functions ---

def format_bytes(byte_value):
    """Formats bytes into KB, MB, GB, etc."""
    if pd.isna(byte_value) or not isinstance(byte_value, (int, float)) or byte_value < 0:
        return "N/A"
    if byte_value < 1024:
        return f"{byte_value} B"
    elif byte_value < 1024**2:
        return f"{byte_value/1024:.2f} KB"
    elif byte_value < 1024**3:
        return f"{byte_value/1024**2:.2f} MB"
    else:
        return f"{byte_value/1024**3:.2f} GB"

def format_bandwidth(bytes_per_second):
    """Formats bytes per second into KB/s, MB/s, etc."""
    if pd.isna(bytes_per_second) or not isinstance(bytes_per_second, (int, float)) or bytes_per_second < 0:
        return "N/A"
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.2f} B/s"
    elif bytes_per_second < 1024**2:
        return f"{bytes_per_second/1024:.2f} KB/s"
    elif bytes_per_second < 1024**3:
        return f"{bytes_per_second/1024**2:.2f} MB/s"
    else:
        return f"{bytes_per_second/1024**3:.2f} GB/s"


def get_table_download_link(df, filename, button_text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    if df is None or df.empty:
        # st.warning(f"Cannot generate download link for {filename}: Data is empty.") # Can be noisy
        return None
    try:
        csv = df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode()).decode() # Not needed for st.download_button data arg
        return st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime='text/csv',
            key=f"download_{filename}" # Add unique key
        )
    except Exception as e:
        st.warning(f"Could not generate table download link for {filename}: {e}")
        logging.exception(f"Error generating CSV download link for {filename}")
        return None


def get_image_download_link(fig, filename, button_text):
    """Generates a link allowing a plotly figure to be downloaded as PNG"""
    if fig is None:
        # st.warning(f"Cannot generate image download link for {filename}: Figure object is None.") # Can be noisy
        return None
    try:
        buffer = BytesIO()
        fig.write_image(buffer, format="png", scale=2) # Increase scale for better resolution
        buffer.seek(0)
        # b64 = base64.b64encode(buffer.read()).decode() # No need to base64 encode for download button data
        return st.download_button(
            label=button_text,
            data=buffer, # Pass the buffer directly
            file_name=filename,
            mime='image/png',
            key=f"download_{filename}" # Add unique key
        )
    except Exception as e:
        st.warning(f"Could not generate image download link for {filename}: {e}")
        logging.exception(f"Error generating PNG download link for {filename}")
        return None

def format_metric_comparison(current_val, baseline_val, higher_is_better=True, is_percentage=False, decimals=2):
    """Formats metric comparison with delta and color coding."""
    # Handle cases where values might be non-numeric or NaN before comparison
    if not isinstance(current_val, (int, float)) or pd.isna(current_val):
        return f"{current_val}", None, "neutral" # Return current value as string if not comparable
    if baseline_val is None or not isinstance(baseline_val, (int, float)) or pd.isna(baseline_val):
        # Format current value even if baseline is not comparable
        current_display = f"{current_val:.{decimals}f}{'%' if is_percentage else ''}"
        return current_display, None, "neutral"

    delta = current_val - baseline_val
    try:
        # Handle division by zero if baseline is 0
        delta_percent = (delta / baseline_val) * 100 if baseline_val != 0 else float('inf') if delta != 0 else 0
    except ZeroDivisionError:
         delta_percent = float('inf') if delta != 0 else 0 # Or set to infinity/NaN representation if preferred

    # Determine color based on whether higher is better
    if higher_is_better:
        if delta > 0.001: # Add tolerance for floating point comparison
            color = "improved"
        elif delta < -0.001:
            color = "degraded"
        else:
            color = "neutral"
    else: # Lower is better (e.g., response time, error rate)
        if delta < -0.001:
            color = "improved"
        elif delta > 0.001:
            color = "degraded"
        else:
            color = "neutral"

    # Format delta display, handle infinite percentage
    delta_percent_str = f"{delta_percent:+.1f}%" if np.isfinite(delta_percent) else "(vs 0)"
    delta_display = f"{delta:+.{decimals}f}{'%' if is_percentage else ''} ({delta_percent_str})"
    current_display = f"{current_val:.{decimals}f}{'%' if is_percentage else ''}"

    return current_display, delta_display, color


# --- Data Loading and Processing ---

def load_test_results(uploaded_file, delimiter=None):
    """Load and preprocess JMeter test results from an uploaded file object"""
    if uploaded_file is None:
        return None
    logging.info(f"Starting processing for file: {uploaded_file.name}")
    try:
        # Read the file content
        file_content = uploaded_file.getvalue()

        # Detect delimiter if not provided
        if delimiter is None:
            try:
                # Decode only the first line for delimiter detection
                first_line = file_content.split(b'\n', 1)[0].decode('utf-8', errors='ignore')
                if '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
                logging.info(f"Detected delimiter: '{repr(delimiter)}' for file '{uploaded_file.name}'")
            except Exception as e:
                 logging.warning(f"Could not detect delimiter for {uploaded_file.name}, defaulting to ','. Error: {e}")
                 delimiter = ','


        # Load the data using BytesIO
        df = pd.read_csv(BytesIO(file_content), delimiter=delimiter, low_memory=False) # low_memory=False can help with mixed types
        logging.info(f"Successfully loaded {len(df)} rows from '{uploaded_file.name}' using delimiter '{repr(delimiter)}'")

        # Check minimum required columns first
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            st.error(f"Missing essential columns in '{uploaded_file.name}': {', '.join(missing_required)}")
            st.info(f"Dashboard requires at least: {', '.join(REQUIRED_COLUMNS)}.")
            return None

        # Check for optional columns and log if missing
        present_columns = df.columns.tolist()
        optional_columns = [col for col in EXPECTED_COLUMNS if col not in REQUIRED_COLUMNS]
        missing_optional = [col for col in optional_columns if col not in present_columns]
        if missing_optional:
            logging.warning(f"Optional columns missing in '{uploaded_file.name}': {', '.join(missing_optional)}. Related metrics/charts may be unavailable.")

        # --- Data Type Conversion and Cleaning ---
        initial_rows = len(df)
        logging.info("Starting data type conversion and cleaning...")

        # timeStamp: Convert epoch milliseconds to datetime
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
        df.dropna(subset=['timeStamp'], inplace=True)
        df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timeStamp'], inplace=True)

        # success: Convert to boolean
        df['success'] = df['success'].astype(str).str.strip().str.lower() == 'true'

        # Numeric columns (handle errors gracefully)
        numeric_cols_optional = ['elapsed', 'Latency', 'Connect', 'bytes', 'sentBytes', 'grpThreads', 'allThreads', 'IdleTime']
        for col in numeric_cols_optional:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Optional: Fill NaNs introduced by coerce with 0 or another value if appropriate
                # df[col].fillna(0, inplace=True)
            else:
                 # Add missing optional numeric columns and fill with NaN or 0
                 df[col] = pd.NA # Use pandas NA for better type handling later
                 # df[col] = 0

        # Drop rows where essential numeric 'elapsed' is NaN after conversion
        df.dropna(subset=['elapsed'], inplace=True)
        # Convert 'elapsed' to int only if not empty and contains valid data
        if not df['elapsed'].empty and df['elapsed'].notna().all():
             try:
                 df['elapsed'] = df['elapsed'].astype(int)
             except pd.errors.IntCastingNaNError:
                 logging.warning("Could not cast 'elapsed' to int due to NaNs after coerce. Keeping as float.")


        # String columns (ensure type and strip whitespace)
        string_cols = ['label', 'responseCode', 'responseMessage', 'failureMessage', 'threadName', 'dataType', 'URL']
        for col in string_cols:
            if col in df.columns:
                 # Convert to string first to handle potential mixed types before strip/fillna
                 df[col] = df[col].astype(str).str.strip().fillna('')
            else:
                 df[col] = '' # Add missing optional string columns as empty strings

        # Log rows dropped during cleaning
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows during cleaning/conversion for '{uploaded_file.name}'.")

        logging.info(f"Finished preprocessing for '{uploaded_file.name}'. Final rows: {len(df)}")
        if len(df) == 0:
             st.warning(f"File '{uploaded_file.name}' resulted in an empty dataset after cleaning. Please check file content.")
             return pd.DataFrame() # Return empty DataFrame instead of None

        return df

    except Exception as e:
        st.error(f"Error processing the file '{uploaded_file.name}': {str(e)}")
        logging.exception(f"Exception during file processing for {uploaded_file.name}") # Log full traceback
        st.info("Please make sure the file is a valid JMeter JTL/CSV file with the expected columns and data types.")
        return None

def calculate_summary_metrics(df):
    """Calculate summary metrics from the JMeter data"""
    if df is None or df.empty: # Check .empty explicitly
        logging.warning("Cannot calculate summary metrics: DataFrame is None or empty.")
        return {}
    logging.info("Calculating summary metrics...")

    metrics = {}
    try:
        # --- Basic Counts ---
        total_samples = len(df)
        successful_samples = df['success'].sum()
        failed_samples = total_samples - successful_samples
        error_rate = (failed_samples / total_samples) * 100 if total_samples > 0 else 0
        metrics.update({
            "Total Samples": total_samples, "Successful Samples": successful_samples,
            "Failed Samples": failed_samples, "Error Rate": error_rate,
        })

        # --- Timing Metrics (Elapsed, Latency, Connect) ---
        for col, name in [('elapsed', 'Response Time'), ('Latency', 'Latency'), ('Connect', 'Connect Time')]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                data_col = df[col].dropna() # Drop NaNs for calculations
                if not data_col.empty:
                     metrics[f"Avg {name}"] = data_col.mean()
                     metrics[f"Median {name}"] = data_col.median()
                     metrics[f"Min {name}"] = data_col.min()
                     metrics[f"Max {name}"] = data_col.max()
                     if col == 'elapsed': # Only calculate percentiles for elapsed time
                          metrics["90th Percentile"] = data_col.quantile(0.90)
                          metrics["95th Percentile"] = data_col.quantile(0.95)
                          metrics["99th Percentile"] = data_col.quantile(0.99)
                          metrics["Std Deviation"] = data_col.std()
                else:
                     logging.warning(f"Column '{col}' is present but empty or all NaN after dropping NaNs.")
            else:
                 logging.warning(f"Timing column '{col}' not found or not numeric. Skipping related metrics.")

        # --- Network Metrics ---
        total_bytes_received = 0
        total_bytes_sent = 0
        if 'bytes' in df.columns and pd.api.types.is_numeric_dtype(df['bytes']):
             metrics["Avg Bytes Received"] = df['bytes'].mean()
             total_bytes_received = df['bytes'].sum()
        if 'sentBytes' in df.columns and pd.api.types.is_numeric_dtype(df['sentBytes']):
             metrics["Avg Bytes Sent"] = df['sentBytes'].mean()
             total_bytes_sent = df['sentBytes'].sum()

        # --- Duration & Throughput ---
        test_start_time = df['timeStamp'].min()
        test_end_time = df['timeStamp'].max()
        test_duration = 0
        requests_per_second = 0
        formatted_duration = "N/A"
        bandwidth_bps = 0 # Bytes per second

        if pd.notna(test_start_time) and pd.notna(test_end_time):
            test_duration = (test_end_time - test_start_time).total_seconds()
            if test_duration > 0:
                 requests_per_second = total_samples / test_duration
                 # Calculate bandwidth using total bytes received over duration
                 if total_bytes_received > 0:
                      bandwidth_bps = total_bytes_received / test_duration
            else:
                 logging.warning(f"Test duration is zero or negative ({test_duration}s). Throughput/Bandwidth set to 0.")

            hours, remainder = divmod(abs(test_duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_duration = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
             logging.warning("Could not determine test duration due to invalid timestamps.")

        metrics.update({
            "Throughput": requests_per_second,
            "Avg Bandwidth (Recv)": bandwidth_bps, # Store raw bytes/sec
            "Test Duration": test_duration,
            "Formatted Duration": formatted_duration,
            "Start Time": test_start_time if pd.notna(test_start_time) else "N/A",
            "End Time": test_end_time if pd.notna(test_end_time) else "N/A"
        })

        # Fill missing metrics with N/A for consistency
        all_possible_metrics = [
             "Total Samples", "Successful Samples", "Failed Samples", "Error Rate",
             "Min Response Time", "Avg Response Time", "Median Response Time", "Max Response Time",
             "90th Percentile", "95th Percentile", "99th Percentile", "Std Deviation",
             "Min Latency", "Avg Latency", "Median Latency", "Max Latency",
             "Min Connect Time", "Avg Connect Time", "Median Connect Time", "Max Connect Time",
             "Avg Bytes Received", "Avg Bytes Sent", "Avg Bandwidth (Recv)",
             "Throughput", "Test Duration", "Formatted Duration", "Start Time", "End Time"
        ]
        for m in all_possible_metrics:
            metrics.setdefault(m, 'N/A') # Set to N/A if not calculated

        logging.info("Summary metrics calculation complete.")
        return metrics
    except Exception as e:
        logging.exception("Error calculating summary metrics")
        st.error(f"An error occurred during summary metrics calculation: {e}")
        return {}


def create_aggregate_report(df):
    """Create an aggregate report table with statistics for each request label"""
    if df is None or df.empty: # Check .empty
        logging.warning("Cannot create aggregate report: DataFrame is None or empty.")
        return None
    logging.info("Creating aggregate report...")

    try:
        # Columns to aggregate
        agg_cols = {
            'elapsed': ['mean', 'min', 'max', 'median', lambda x: x.quantile(0.90), lambda x: x.quantile(0.95), lambda x: x.quantile(0.99), 'std'],
            'success': ['count', 'sum'], # count is total samples, sum is successful samples
        }
        # Add optional columns if they exist and are numeric
        if 'Latency' in df.columns and pd.api.types.is_numeric_dtype(df['Latency']):
             agg_cols['Latency'] = ['mean', 'median', 'max']
        if 'Connect' in df.columns and pd.api.types.is_numeric_dtype(df['Connect']):
             agg_cols['Connect'] = ['mean', 'median', 'max']
        if 'bytes' in df.columns and pd.api.types.is_numeric_dtype(df['bytes']):
             agg_cols['bytes'] = ['mean', 'sum']
        if 'sentBytes' in df.columns and pd.api.types.is_numeric_dtype(df['sentBytes']):
             agg_cols['sentBytes'] = ['mean', 'sum']

        # Group and aggregate
        grouped = df.groupby('label', observed=False) if pd.__version__ >= '1.5.0' else df.groupby('label')
        agg_df = grouped.agg(agg_cols)

        # Flatten MultiIndex columns and rename
        agg_df.columns = ['_'.join(map(str, col)).strip('_') for col in agg_df.columns.values] # Ensure col parts are strings, strip trailing _ if lambda name is empty
        rename_map = {
            'elapsed_<lambda_0>': '90% (ms)', 'elapsed_<lambda_1>': '95% (ms)', 'elapsed_<lambda_2>': '99% (ms)',
            'elapsed_mean': 'Avg (ms)', 'elapsed_min': 'Min (ms)', 'elapsed_max': 'Max (ms)',
            'elapsed_median': 'Median (ms)', 'elapsed_std': 'StdDev',
            'success_count': 'Samples', 'success_sum': 'Successful',
            'Latency_mean': 'Avg Latency (ms)', 'Latency_median': 'Median Latency (ms)', 'Latency_max': 'Max Latency (ms)',
            'Connect_mean': 'Avg Connect (ms)', 'Connect_median': 'Median Connect (ms)', 'Connect_max': 'Max Connect (ms)',
            'bytes_mean': 'Avg Bytes Recv', 'bytes_sum': 'Total Bytes Recv',
            'sentBytes_mean': 'Avg Bytes Sent', 'sentBytes_sum': 'Total Bytes Sent'
        }
        # Only rename columns that actually exist after aggregation
        agg_df.rename(columns={k: v for k, v in rename_map.items() if k in agg_df.columns}, inplace=True)


        # Calculate derived metrics
        agg_df['Errors'] = agg_df['Samples'] - agg_df['Successful']
        agg_df['Error %'] = (agg_df['Errors'] / agg_df['Samples'] * 100).fillna(0)

        # Calculate throughput per label
        def calc_throughput(group):
            total = len(group)
            throughput = 0
            if total > 1:
                min_time = group['timeStamp'].min()
                max_time = group['timeStamp'].max()
                if pd.notna(min_time) and pd.notna(max_time):
                    duration = (max_time - min_time).total_seconds()
                    if duration > 0:
                        throughput = total / duration
            return throughput

        throughput_series = grouped.apply(calc_throughput)
        agg_df['Throughput (/sec)'] = throughput_series

        # Reset index to make 'label' a column
        agg_df = agg_df.reset_index()
        # FIX: Rename 'label' column to 'Label' (uppercase)
        agg_df.rename(columns={'label': 'Label'}, inplace=True)


        # Select and reorder columns
        final_cols = [
            'Label', 'Samples', 'Errors', 'Error %',
            'Avg (ms)', 'Median (ms)', '90% (ms)', '95% (ms)', '99% (ms)', 'Min (ms)', 'Max (ms)', 'StdDev',
            'Avg Latency (ms)', 'Median Latency (ms)', 'Max Latency (ms)', # Optional Latency
            'Avg Connect (ms)', 'Median Connect (ms)', 'Max Connect (ms)', # Optional Connect
            'Avg Bytes Recv', 'Avg Bytes Sent', # Optional Bytes
            'Throughput (/sec)'
        ]
        # Keep only columns that actually exist in the aggregated df
        final_cols = [col for col in final_cols if col in agg_df.columns]
        agg_df = agg_df[final_cols]

        # Fill any remaining NaNs (e.g., StdDev for single sample) with 0
        agg_df.fillna(0, inplace=True)

        agg_df.sort_values(by='Label', inplace=True) # Now sorts by 'Label' (uppercase)
        logging.info("Aggregate report creation complete.")
        return agg_df
    except Exception as e:
        logging.exception("Error creating aggregate report")
        st.error(f"An error occurred during aggregate report creation: {e}")
        return None


# --- Display Functions ---

def display_summary_metrics(metrics, comparison_metrics=None):
    """Display summary metrics with optional comparison using metric cards."""
    st.markdown("<h2 class='sub-header'>Summary Metrics</h2>", unsafe_allow_html=True)

    if not metrics:
        st.warning("No summary metrics to display.")
        return

    # --- Main Metric Cards ---
    # Define metrics to display in cards, checking if they exist in the calculated metrics
    metrics_in_cards_config = [
        ("Total Samples", "", True, False, 0),
        ("Avg Response Time", " ms", False, False, 2),
        ("Avg Latency", " ms", False, False, 2), # New
        ("Avg Connect Time", " ms", False, False, 2), # New
        ("Error Rate", "%", False, True, 2),
        ("Throughput", " req/sec", True, False, 2),
        ("Avg Bandwidth (Recv)", "", True, False, 2), # New - unit added in formatting
        ("95th Percentile", " ms", False, False, 2), # Keep 95th as key percentile
    ]

    # Filter config based on available metrics
    metrics_in_cards = [(label, unit, higher_is_better, is_percentage, decimals)
                        for label, unit, higher_is_better, is_percentage, decimals in metrics_in_cards_config
                        if metrics.get(label, 'N/A') != 'N/A'] # Only show if metric was calculated

    num_cards = len(metrics_in_cards)
    num_cols = 4 # Desired columns
    cols = st.columns(num_cols)

    card_col_index = 0
    for label, unit, higher_is_better, is_percentage, decimals in metrics_in_cards:
         with cols[card_col_index % num_cols]:
              value = metrics.get(label, 'N/A')
              comp_value = comparison_metrics.get(label) if comparison_metrics else None

              # Use helper to format and display
              formatted_value = "N/A"
              delta_display = None
              color_class = "neutral"

              # Special formatting for bandwidth
              is_bandwidth = label == "Avg Bandwidth (Recv)"
              if is_bandwidth and isinstance(value, (int, float)):
                   formatted_value = format_bandwidth(value)
                   # Comparison for bandwidth (higher is better)
                   if comp_value is not None and comp_value != 'N/A':
                        try:
                             numeric_comp_bw = float(comp_value)
                             _, delta_display_raw, color_class = format_metric_comparison(value, numeric_comp_bw, True, False, 2)
                             # Overwrite delta to show formatted bytes/sec diff
                             delta_bw = value - numeric_comp_bw
                             delta_percent_bw = (delta_bw / numeric_comp_bw * 100) if numeric_comp_bw != 0 else float('inf')
                             delta_percent_bw_str = f"{delta_percent_bw:+.1f}%" if np.isfinite(delta_percent_bw) else "(vs 0)"
                             # Format the absolute delta using bandwidth formatter
                             formatted_delta_bw = f"+{format_bandwidth(delta_bw)}" if delta_bw >=0 else f"-{format_bandwidth(abs(delta_bw))}"
                             delta_display = f"{formatted_delta_bw} ({delta_percent_bw_str})"

                        except (ValueError, TypeError): delta_display = None # Ignore comparison if baseline not numeric
                   else: delta_display = None # No comparison if baseline N/A
              # Standard formatting for other metrics
              elif value is not None and value != "N/A":
                   if isinstance(value, datetime): # Should not happen for these metrics
                        formatted_value = value.strftime('%Y-%m-%d %H:%M:%S')
                   else:
                        try:
                             numeric_value = float(value)
                             numeric_comparison = None
                             if comp_value is not None and comp_value != "N/A":
                                  try: numeric_comparison = float(comp_value)
                                  except (ValueError, TypeError): pass

                             formatted_value, delta_display, color_class = format_metric_comparison(
                                  numeric_value, numeric_comparison, higher_is_better, is_percentage, decimals
                             )
                             # Apply unit only if formatting was successful
                             formatted_value = f"{formatted_value}{unit}"
                        except (ValueError, TypeError):
                             formatted_value = f"{value}{unit}" # Display as is if not numeric

              # Display inside styled container
              with st.container(border=True):
                   st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
                   st.markdown(f"<div class='metric-value'>{formatted_value}</div>", unsafe_allow_html=True)
                   if delta_display:
                        st.markdown(f"<div class='metric-delta {color_class}-value'>{delta_display}</div>", unsafe_allow_html=True)

         card_col_index += 1

    # --- Duration/Start/End Time Row ---
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True) # Add some space
    cols_time = st.columns(3)
    with cols_time[0]:
         with st.container(border=False): # No border for this row
              st.markdown("<div class='time-info-metric'>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-label'>Duration</div>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-value'>{metrics.get('Formatted Duration', 'N/A')}</div>", unsafe_allow_html=True)
              st.markdown("</div>", unsafe_allow_html=True)
    with cols_time[1]:
         start_time = metrics.get('Start Time', 'N/A')
         start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_time, datetime) else str(start_time)
         with st.container(border=False):
              st.markdown("<div class='time-info-metric'>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-label'>Started</div>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-value'>{start_time_str}</div>", unsafe_allow_html=True)
              st.markdown("</div>", unsafe_allow_html=True)
    with cols_time[2]:
         end_time = metrics.get('End Time', 'N/A')
         end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_time, datetime) else str(end_time)
         with st.container(border=False):
              st.markdown("<div class='time-info-metric'>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-label'>Ended</div>", unsafe_allow_html=True)
              st.markdown(f"<div class='metric-value'>{end_time_str}</div>", unsafe_allow_html=True)
              st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")


def display_aggregate_report(agg_df, comparison_agg_df=None):
    """Display the aggregate report with optional comparison, handling numeric conversions."""
    st.markdown("<h2 class='sub-header'>Aggregate Report by Label</h2>", unsafe_allow_html=True)

    if agg_df is None or agg_df.empty: # Check .empty
        st.warning("No data available for aggregate report.")
        return

    # Define columns to format and potentially compare
    # Prioritize key performance indicators
    cols_to_display_base = [
        'Label', 'Samples', 'Errors', 'Error %',
        'Avg (ms)', 'Median (ms)', '90% (ms)', '95% (ms)', 'Max (ms)',
        'Throughput (/sec)'
    ]
    # Add optional columns if they exist in the dataframe
    optional_cols_display = [
        'Avg Latency (ms)', 'Avg Connect (ms)', 'Avg Bytes Recv', 'Avg Bytes Sent'
    ]
    cols_to_display = cols_to_display_base + [col for col in optional_cols_display if col in agg_df.columns]

    # Select only the columns we intend to display
    # Ensure 'Label' exists before selecting
    if 'Label' not in agg_df.columns:
         st.error("Internal Error: 'Label' column missing in aggregate report after processing.")
         logging.error("'Label' column missing from agg_df before selection.")
         return
    # Filter cols_to_display to only include columns actually present in agg_df
    cols_to_display = [col for col in cols_to_display if col in agg_df.columns]
    display_df_raw = agg_df[cols_to_display].copy()


    # Format columns for the display dataframe (non-comparison view)
    display_df_formatted = display_df_raw.copy()
    for col in display_df_formatted.columns:
        if col == 'Label': continue
        try:
            if '(ms)' in col:
                display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            elif 'Error %' in col:
                display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            elif 'Bytes' in col:
                 display_df_formatted[col] = display_df_formatted[col].apply(lambda x: format_bytes(x) if pd.notna(x) else "N/A")
            elif 'Throughput' in col:
                 display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            elif col in ['Samples', 'Errors']:
                 display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int, float)) else ("N/A" if pd.isna(x) else str(x)))
        except Exception as e:
             logging.warning(f"Could not format column '{col}' in aggregate report: {e}")
             display_df_formatted[col] = display_df_formatted[col].astype(str) # Fallback


    if comparison_agg_df is not None and not comparison_agg_df.empty: # Check if baseline df is valid
        st.info("Comparing current results with baseline.")
        # Ensure comparison dataframe has same columns for merging
        comparison_agg_df_copy = comparison_agg_df.copy()
        # Ensure 'Label' column exists in baseline
        if 'Label' not in comparison_agg_df_copy.columns:
             if 'label' in comparison_agg_df_copy.columns:
                  comparison_agg_df_copy.rename(columns={'label': 'Label'}, inplace=True)
             else:
                  st.warning("Baseline aggregate report missing 'Label' column. Cannot perform comparison.")
                  comparison_agg_df = None # Disable comparison

    # Proceed with comparison only if baseline is still valid
    if comparison_agg_df is not None and not comparison_agg_df.empty:
        try:
            # Merge dataframes on Label using only columns intended for display
            merged_df = pd.merge(display_df_raw, comparison_agg_df_copy, on='Label', suffixes=('_Current', '_Baseline'), how='outer')

            # Fill NaNs introduced by outer merge
            numeric_cols_for_comp = [col for col in cols_to_display if col != 'Label'] # Get numeric cols being displayed
            num_cols_to_fill = [f"{col}_{suffix}" for col in numeric_cols_for_comp for suffix in ['Current', 'Baseline']]
            fill_values = {col: 0 for col in num_cols_to_fill if col in merged_df.columns} # Fill numeric NaNs with 0 for calc
            merged_df.fillna(fill_values, inplace=True)
            merged_df.fillna({'Label': 'Unknown'}, inplace=True) # Handle potential NaN labels

            # Prepare dataframe for display with comparison columns
            display_comparison_df = pd.DataFrame()
            display_comparison_df['Label'] = merged_df['Label']

            # Columns to show in comparison view (subset of displayed cols)
            compare_cols_subset = ['Samples', 'Error %', 'Avg (ms)', '95% (ms)', 'Throughput (/sec)', 'Avg Latency (ms)', 'Avg Bytes Recv']

            # Add columns with comparison formatting
            for col_name in compare_cols_subset:
                current_col = f"{col_name}_Current"
                baseline_col = f"{col_name}_Baseline"

                # Check if columns exist (they might not if optional columns weren't present)
                if current_col in merged_df.columns and baseline_col in merged_df.columns:
                    higher_is_better = col_name not in ['Error %', 'Avg (ms)', '95% (ms)', 'Avg Latency (ms)']
                    is_percentage = col_name == 'Error %'
                    is_bytes = 'Bytes' in col_name
                    decimals = 0 if col_name == 'Samples' else 2

                    # Apply comparison formatting row-wise
                    comparison_results = merged_df.apply(
                        lambda row: format_metric_comparison(
                            row[current_col], row[baseline_col], higher_is_better, is_percentage, decimals
                        ), axis=1
                    )

                    display_comparison_df[f'{col_name} (Current)'] = [res[0] for res in comparison_results]
                    # Format baseline value separately
                    display_comparison_df[f'{col_name} (Baseline)'] = merged_df[baseline_col].apply(
                        lambda x: format_bytes(x) if is_bytes and pd.notna(x) else (f"{x:.{decimals}f}{'%' if is_percentage else ''}" if pd.notna(x) and isinstance(x, (int, float)) else ("N/A" if pd.isna(x) else str(x)))
                    )
                    display_comparison_df[f'{col_name} (Delta)'] = [res[1] for res in comparison_results]

                    # Re-add units/formatting to current column after comparison calc
                    if is_bytes: display_comparison_df[f'{col_name} (Current)'] = merged_df[current_col].apply(lambda x: format_bytes(x) if pd.notna(x) else "N/A")
                    elif '(ms)' in col_name: display_comparison_df[f'{col_name} (Current)'] += ' ms'
                    elif 'Throughput' in col_name: display_comparison_df[f'{col_name} (Current)'] += ' /sec'
                    elif col_name == 'Samples': display_comparison_df[f'{col_name} (Current)'] = merged_df[current_col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")


            # Select and reorder columns for display
            display_cols_order = ['Label']
            for col_name in compare_cols_subset:
                 # Check if comparison columns were successfully created
                 if f'{col_name} (Current)' in display_comparison_df.columns:
                      display_cols_order.extend([f'{col_name} (Current)', f'{col_name} (Baseline)', f'{col_name} (Delta)'])

            # Filter out columns that might not exist
            display_cols_order = [col for col in display_cols_order if col in display_comparison_df.columns]

            st.dataframe(display_comparison_df[display_cols_order], use_container_width=True)
            get_table_download_link(display_comparison_df[display_cols_order], "aggregate_report_comparison.csv", "Download Comparison Report (CSV)")

        except Exception as e:
             st.error(f"Error occurred during aggregate report comparison: {e}")
             logging.exception("Error during aggregate comparison processing")
             st.dataframe(display_df_formatted, use_container_width=True) # Fallback to showing current formatted only
             get_table_download_link(agg_df, "aggregate_report.csv", "Download Raw Report (CSV)")

    else:
        # Single test report - show formatted data
        st.dataframe(display_df_formatted, use_container_width=True)
        get_table_download_link(agg_df, "aggregate_report.csv", "Download Raw Report (CSV)") # Download raw numeric data

    st.markdown("---")


def display_response_time_charts(df, comparison_df=None):
    """Display response time analysis charts with optional comparison"""
    st.markdown("<h2 class='sub-header'>Response Time Analysis</h2>", unsafe_allow_html=True)

    if df is None or df.empty: # Check .empty
        st.warning("No data available for response time analysis.")
        return

    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Response Time Trend",
        "📊 Response Time Distribution",
        "📉 Response Time Percentiles",
        "🏷️ Response Time by Label"
    ])

    # --- Tab 1: Response Time Trend ---
    with tab1:
        st.markdown("#### Response Time Over Time")
        try:
            # Prepare data for time series
            df_time = df.copy()
            df_time.set_index('timeStamp', inplace=True)

            # Determine resampling rate based on test duration
            test_duration_sec = (df['timeStamp'].max() - df['timeStamp'].min()).total_seconds()
            if test_duration_sec <= 0: # Handle zero duration case
                 resample_rate = '1s'
                 resample_label = '1 second'
                 logging.warning("Test duration is zero or negative, using 1s resampling for trend.")
            elif test_duration_sec <= 600: # Up to 10 mins
                 resample_rate = '5s'
                 resample_label = '5 seconds'
            elif test_duration_sec <= 3600: # Up to 1 hour
                 resample_rate = '15s'
                 resample_label = '15 seconds'
            elif test_duration_sec <= 3 * 3600: # Up to 3 hours
                 resample_rate = '30s'
                 resample_label = '30 seconds'
            else: # Over 3 hours
                 resample_rate = '1min'
                 resample_label = '1 minute'

            # Resample data
            df_resampled = df_time.resample(resample_rate).agg({
                'elapsed': ['mean', 'median', lambda x: x.quantile(0.95), 'max'] # Mean, Median, 95th Pctl, Max
            }).dropna() # Drop intervals with no data
            df_resampled.columns = ['Mean', 'Median', 'P95', 'Max'] # Simpler column names
            df_resampled = df_resampled.reset_index()

            # Create time series chart with plotly
            fig_trend = go.Figure()

            # Add traces for current data if resampled data is not empty
            if not df_resampled.empty:
                fig_trend.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Mean'], mode='lines', name='Mean (Current)', line=dict(color='royalblue')))
                fig_trend.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Median'], mode='lines', name='Median (Current)', line=dict(color='forestgreen')))
                fig_trend.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['P95'], mode='lines', name='95th Pctl (Current)', line=dict(color='darkorange')))
                fig_trend.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Max'], mode='lines', name='Max (Current)', line=dict(color='firebrick', dash='dot')))
            else:
                 st.warning("No data points available for Response Time Trend after resampling.")


            # Add comparison data if available
            if comparison_df is not None and not comparison_df.empty: # Check .empty
                comp_df_time = comparison_df.copy()
                comp_df_time.set_index('timeStamp', inplace=True)
                # Resample comparison data using the same rate
                comp_df_resampled = comp_df_time.resample(resample_rate).agg({'elapsed': ['mean', 'median']}).dropna() # Only show mean/median for baseline for clarity
                comp_df_resampled.columns = ['Mean', 'Median']
                comp_df_resampled = comp_df_resampled.reset_index()

                # Add comparison traces if resampled data is not empty
                if not comp_df_resampled.empty:
                    fig_trend.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Mean'], mode='lines', name='Mean (Baseline)', line=dict(color='lightblue', dash='dash')))
                    fig_trend.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Median'], mode='lines', name='Median (Baseline)', line=dict(color='lightgreen', dash='dash')))

            fig_trend.update_layout(
                title=f'Response Times Over Time (Aggregated every {resample_label})',
                xaxis_title='Time',
                yaxis_title='Response Time (ms)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            get_image_download_link(fig_trend, "response_time_trend.png", "Download Trend Chart (PNG)")
        except Exception as e:
            st.error(f"Could not generate Response Time Trend chart: {e}")
            logging.exception("Error generating Response Time Trend chart")

    # --- Tab 2: Response Time Distribution ---
    with tab2:
        st.markdown("#### Response Time Distribution (Box Plot)")
        try:
            # Use Plotly for better interactivity and comparison handling
            fig_dist = go.Figure()
            unique_labels_current = df['label'].unique()

            # Add current data
            for i, label in enumerate(unique_labels_current):
                fig_dist.add_trace(go.Box(
                    y=df[df['label'] == label]['elapsed'],
                    name=f"{label} (Current)",
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                ))

            # Add comparison data
            if comparison_df is not None and not comparison_df.empty: # Check .empty
                 unique_labels_baseline = comparison_df['label'].unique()
                 for i, label in enumerate(unique_labels_baseline):
                     # Check if label exists in current data for color consistency (optional)
                     try:
                         current_label_index = list(unique_labels_current).index(label)
                         color = px.colors.qualitative.Plotly[current_label_index % len(px.colors.qualitative.Plotly)]
                     except ValueError:
                         color = px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)] # Use a different palette if label is new

                     # FIX: Removed invalid 'line=dict(dash='dash')' and added marker opacity
                     fig_dist.add_trace(go.Box(
                         y=comparison_df[comparison_df['label'] == label]['elapsed'],
                         name=f"{label} (Baseline)",
                         marker_color=color,
                         marker=dict(opacity=0.6) # Add opacity for visual distinction
                         # line=dict(dash='dash') # REMOVED: Invalid property
                     ))

            fig_dist.update_layout(
                title='Response Time Distribution by Request Type',
                yaxis_title='Response Time (ms)',
                xaxis_title='Request Label',
                height=500,
                boxmode='group', # Group boxes side-by-side if comparing
                showlegend=True # Show legend especially when comparing
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            get_image_download_link(fig_dist, "response_time_distribution.png", "Download Distribution Chart (PNG)")
        except Exception as e:
            st.error(f"Could not generate Response Time Distribution chart: {e}")
            logging.exception("Error generating Response Time Distribution chart")


    # --- Tab 3: Response Time Percentiles ---
    with tab3:
        st.markdown("#### Overall Response Time Percentiles")
        try:
            percentiles = np.arange(0, 101, 5) # 0, 5, 10, ..., 95, 100
            percentiles = np.append(percentiles, [99, 99.9]) # Add 99 and 99.9
            percentiles = np.sort(np.unique(percentiles))

            # Calculate percentiles for current test
            current_percentile_values = np.percentile(df['elapsed'], percentiles)

            # Create percentile chart
            fig_perc = go.Figure()

            fig_perc.add_trace(go.Scatter(
                x=percentiles,
                y=current_percentile_values,
                mode='lines+markers',
                name='Current Test',
                line=dict(color='blue', width=2)
            ))

            # Calculate and add percentiles for comparison test if available
            if comparison_df is not None and not comparison_df.empty: # Check .empty
                comparison_percentile_values = np.percentile(comparison_df['elapsed'], percentiles)
                fig_perc.add_trace(go.Scatter(
                    x=percentiles,
                    y=comparison_percentile_values,
                    mode='lines+markers',
                    name='Baseline Test',
                    line=dict(color='red', width=2, dash='dash')
                ))

            fig_perc.update_layout(
                title='Overall Response Time Percentiles',
                xaxis_title='Percentile',
                yaxis_title='Response Time (ms)',
                height=500,
                xaxis=dict(
                    tickvals=np.arange(0, 101, 10), # Major ticks every 10
                    ticktext=[f"{p}%" for p in np.arange(0, 101, 10)]
                ),
                yaxis=dict(type='log') # Log scale often useful for percentiles
            )
            st.plotly_chart(fig_perc, use_container_width=True)
            st.caption("Note: Y-axis is logarithmic for better visualization of higher percentiles.")
            get_image_download_link(fig_perc, "response_time_percentiles.png", "Download Percentiles Chart (PNG)")
        except Exception as e:
            st.error(f"Could not generate Response Time Percentiles chart: {e}")
            logging.exception("Error generating Response Time Percentiles chart")


    # --- Tab 4: Response Time by Label (Bar Chart) ---
    with tab4:
        st.markdown("#### Key Response Time Metrics by Label")
        try:
            agg_report = create_aggregate_report(df) # Use the aggregate report function

            if agg_report is not None and not agg_report.empty: # Check .empty
                # Select key metrics to plot
                metrics_to_plot = ['Avg (ms)', 'Median (ms)', '90% (ms)', '95% (ms)', '99% (ms)']
                # Ensure Label column exists before proceeding
                if 'Label' not in agg_report.columns:
                     st.error("Internal error: 'Label' column missing in aggregate report for chart.")
                     logging.error("Label column missing in agg_report for RT by Label chart.")
                else:
                    plot_data = agg_report[['Label'] + [col for col in metrics_to_plot if col in agg_report.columns]] # Select existing metrics
                    plot_data = plot_data.sort_values(by='Label') # Sort for consistent y-axis order

                    # Create grouped HORIZONTAL bar chart
                    fig_label = go.Figure()
                    num_metrics = len(metrics_to_plot)

                    # Add current data bars (Horizontal)
                    for i, metric in enumerate(metrics_to_plot):
                         if metric in plot_data.columns: # Check if metric exists
                             fig_label.add_trace(go.Bar(
                                 y=plot_data['Label'], # Labels on Y-axis
                                 x=plot_data[metric],  # Values on X-axis
                                 name=f"{metric} (Current)",
                                 marker_color=px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)],
                                 orientation='h' # Set orientation to horizontal
                             ))

                    # Add comparison data bars if available
                    if comparison_df is not None and not comparison_df.empty: # Check .empty
                        comp_agg_report = create_aggregate_report(comparison_df)
                        if comp_agg_report is not None and not comp_agg_report.empty and 'Label' in comp_agg_report.columns: # Check .empty and Label
                            comp_plot_data = comp_agg_report[['Label'] + [col for col in metrics_to_plot if col in comp_agg_report.columns]]
                            # Merge to align labels properly
                            merged_plot_data = pd.merge(plot_data[['Label']], # Only need labels from current for merging structure
                                                        comp_plot_data,
                                                        on='Label', how='left') # Left merge to keep order of current labels
                            # Fill NaN for metrics where label didn't exist in baseline report
                            merged_plot_data.fillna({m: 0 for m in metrics_to_plot}, inplace=True)
                            # Sort merged data to match plot_data order (important for y-axis alignment)
                            merged_plot_data = pd.merge(plot_data[['Label']], merged_plot_data, on='Label', how='left')


                            for i, metric in enumerate(metrics_to_plot):
                                 baseline_col = metric # Column name is just metric in merged_plot_data now
                                 if baseline_col in merged_plot_data.columns:
                                     fig_label.add_trace(go.Bar(
                                         y=merged_plot_data['Label'], # Labels on Y-axis
                                         x=merged_plot_data[baseline_col], # Values on X-axis
                                         name=f"{metric} (Baseline)",
                                         marker_color=px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)],
                                         opacity=0.5, # Make baseline bars semi-transparent
                                         orientation='h' # Set orientation to horizontal
                                     ))

                    # Determine appropriate height based on number of labels
                    num_labels = len(plot_data['Label'])
                    chart_height = max(500, num_labels * 25) # Base height 500, add 25px per label

                    fig_label.update_layout(
                        title='Response Time Statistics by Request Label',
                        yaxis_title='Request Label', # Swapped
                        xaxis_title='Response Time (ms)', # Swapped
                        barmode='group', # Group bars by label
                        height=chart_height, # Dynamic height
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        # Set y-axis category order based on the sorted plot_data labels
                        yaxis={'categoryorder':'array', 'categoryarray':plot_data['Label'].tolist(), 'autorange': 'reversed'}, # Reversed often looks better for horizontal
                        margin=dict(l=200, t=50) # Increase left margin significantly for long labels, add top margin
                    )
                    st.plotly_chart(fig_label, use_container_width=True)
                    get_image_download_link(fig_label, "response_time_by_label.png", "Download Label Metrics Chart (PNG)")
            else:
                st.warning("No data available in aggregate report for label chart.")
        except Exception as e:
            st.error(f"Could not generate Response Time by Label chart: {e}")
            logging.exception("Error generating Response Time by Label chart")

    st.markdown("---")


def display_throughput_charts(df, comparison_df=None):
    """Display throughput analysis charts."""
    st.markdown("<h2 class='sub-header'>Throughput Analysis</h2>", unsafe_allow_html=True)

    if df is None or df.empty: # Check .empty
        st.warning("No data available for throughput analysis.")
        return

    tab1, tab2 = st.tabs(["📈 Throughput Over Time", "🏷️ Throughput by Label"])

    with tab1:
        st.markdown("#### Requests Per Second Over Time")
        try:
            df_time = df.copy()
            df_time.set_index('timeStamp', inplace=True)

            # Resample to get counts per second
            df_resampled = df_time.resample('1s').size().reset_index(name='RequestsPerSecond')

            # Create time series chart
            fig_tps = go.Figure()
            if not df_resampled.empty:
                fig_tps.add_trace(go.Scatter(
                    x=df_resampled['timeStamp'],
                    y=df_resampled['RequestsPerSecond'],
                    mode='lines',
                    name='Requests/sec (Current)',
                    line=dict(color='purple')
                ))
            else:
                 st.warning("No data points available for Throughput Over Time after resampling.")


            # Add comparison data if available
            if comparison_df is not None and not comparison_df.empty: # Check .empty
                comp_df_time = comparison_df.copy()
                comp_df_time.set_index('timeStamp', inplace=True)
                comp_df_resampled = comp_df_time.resample('1s').size().reset_index(name='RequestsPerSecond')
                if not comp_df_resampled.empty:
                    fig_tps.add_trace(go.Scatter(
                        x=comp_df_resampled['timeStamp'],
                        y=comp_df_resampled['RequestsPerSecond'],
                        mode='lines',
                        name='Requests/sec (Baseline)',
                        line=dict(color='plum', dash='dash')
                    ))

            fig_tps.update_layout(
                title='Throughput (Requests per Second) Over Time',
                xaxis_title='Time',
                yaxis_title='Requests per Second',
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig_tps, use_container_width=True)
            get_image_download_link(fig_tps, "throughput_trend.png", "Download Throughput Trend (PNG)")
        except Exception as e:
            st.error(f"Could not generate Throughput Over Time chart: {e}")
            logging.exception("Error generating Throughput Over Time chart")


    with tab2:
        st.markdown("#### Average Throughput by Label")
        try:
            agg_report = create_aggregate_report(df) # Re-use aggregate report

            if agg_report is not None and not agg_report.empty and 'Label' in agg_report.columns: # Check .empty and Label
                 # Create bar chart for throughput by label
                 fig_label_tps = go.Figure()
                 fig_label_tps.add_trace(go.Bar(
                     x=agg_report['Label'],
                     y=agg_report['Throughput (/sec)'],
                     name='Throughput (Current)',
                     marker_color='teal'
                 ))

                 # Add comparison data
                 if comparison_df is not None and not comparison_df.empty: # Check .empty
                     comp_agg_report = create_aggregate_report(comparison_df)
                     if comp_agg_report is not None and not comp_agg_report.empty and 'Label' in comp_agg_report.columns: # Check .empty and Label
                         # Merge to align labels
                         merged_tps = pd.merge(agg_report[['Label', 'Throughput (/sec)']],
                                               comp_agg_report[['Label', 'Throughput (/sec)']],
                                               on='Label', suffixes=('_Current', '_Baseline'), how='outer')
                         # Fill NaN for metrics where label didn't exist in one report
                         merged_tps.fillna({'Throughput (/sec)_Baseline': 0}, inplace=True)

                         fig_label_tps.add_trace(go.Bar(
                             x=merged_tps['Label'],
                             y=merged_tps['Throughput (/sec)_Baseline'],
                             name='Throughput (Baseline)',
                             marker_color='cyan',
                             opacity=0.6
                         ))

                 fig_label_tps.update_layout(
                     title='Average Throughput by Request Label',
                     xaxis_title='Request Label',
                     yaxis_title='Average Throughput (Requests/sec)',
                     barmode='group',
                     height=400,
                     xaxis={'categoryorder':'array', 'categoryarray':sorted(agg_report['Label'].tolist())}
                 )
                 # Rotate labels if many labels exist
                 if len(agg_report['Label']) > 15:
                      fig_label_tps.update_layout(xaxis_tickangle=-45)

                 st.plotly_chart(fig_label_tps, use_container_width=True)
                 get_image_download_link(fig_label_tps, "throughput_by_label.png", "Download Label Throughput Chart (PNG)")
            else:
                 st.warning("Could not generate aggregate report for label throughput chart.")
        except Exception as e:
            st.error(f"Could not generate Throughput by Label chart: {e}")
            logging.exception("Error generating Throughput by Label chart")


    st.markdown("---")

# Helper function to categorize response codes
def get_code_category(code):
    """Categorizes HTTP response codes (handles floats like 400.0)."""
    try:
        if pd.isna(code):
            return "Unknown"

        # Handle float-like numbers (e.g., 400.0) or strings like "400.0"
        code_str = str(code).strip()
        if '.' in code_str:
            code_str = code_str.split('.')[0]  # Get the integer part

        if re.match(r"^[1-5]\d{2}$", code_str):
            code_int = int(code_str)
            if 100 <= code_int < 200: return "1xx Informational"
            if 200 <= code_int < 300: return "2xx Success"
            if 300 <= code_int < 400: return "3xx Redirection"
            if 400 <= code_int < 500: return "4xx Client Error"
            if 500 <= code_int < 600: return "5xx Server Error"
        elif "non http" in code_str.lower():
            return "Non HTTP"
        else:
            return "Other"
    except Exception:
        return "Unknown"

def display_error_analysis(df, comparison_df=None):
    """Display error analysis: summary, breakdown by type and label."""
    st.markdown("<h2 class='sub-header'>Error Analysis</h2>", unsafe_allow_html=True)

    if df is None or df.empty: # Check .empty
        st.warning("No data available for error analysis.")
        return

    try:
        errors_df = df[~df['success']].copy()
        total_errors = len(errors_df)
        total_samples = len(df)
        error_rate = (total_errors / total_samples) * 100 if total_samples > 0 else 0

        st.metric("Total Errors", f"{total_errors:,} ({error_rate:.2f}%)")

        if total_errors == 0:
            st.success("🎉 No errors detected in this test run!")
            st.markdown("---")
            return

        # Error Breakdown Table
        st.markdown("#### Error Breakdown by Label & Message")
        # Use the dedicated function which handles missing columns etc.
        error_summary = create_error_breakdown_table(df) # Includes Label, Code, Message, Count, %

        if error_summary is not None and not error_summary.empty:
             # Check if it's the special error message DataFrame
             if 'Error' in error_summary.columns and len(error_summary) == 1:
                  st.error(f"Could not generate error breakdown table: {error_summary['Error'].iloc[0]}")
             else:
                  # Add URL if available
                  error_summary_display = error_summary.copy() # Start with the summary
                  if 'URL' in df.columns:
                       try:
                            # Prepare grouping keys, handle missing responseMessage
                            group_keys = ['label', 'responseCode']
                            # Check if responseMessage exists in BOTH dataframes before adding
                            if 'responseMessage' in df.columns and 'responseMessage' in error_summary_display.columns:
                                 group_keys.append('responseMessage')
                            # Get first URL for each error group
                            url_map = df[~df['success']].groupby(group_keys)['URL'].first().reset_index()
                            # Merge URL back into the summary
                            error_summary_display = pd.merge(error_summary_display, url_map, on=group_keys, how='left')
                            # Truncate long URLs for display
                            error_summary_display['URL'] = error_summary_display['URL'].astype(str).str.slice(0, 100) + '...'
                       except Exception as merge_err:
                            logging.warning(f"Could not merge URL into error breakdown: {merge_err}")
                            # Continue without URL column if merge fails

                  st.dataframe(error_summary_display, use_container_width=True)
                  get_table_download_link(error_summary_display, "error_breakdown.csv", "Download Error Breakdown (CSV)")
        else:
             st.warning("Could not generate error breakdown table (no errors found or an issue occurred).")


        # --- Charts ---
        col1, col2 = st.columns(2)

        with col1:
            # Errors by Label (Pie Chart)
            st.markdown("#### Errors by Label")
            try:
                if not errors_df.empty:
                    errors_by_label = errors_df['label'].value_counts().reset_index()
                    errors_by_label.columns = ['Label', 'Count']
                    fig_err_label = px.pie(errors_by_label, values='Count', names='Label',
                                           title='Error Distribution by Request Label',
                                           hole=0.3) # Donut chart
                    fig_err_label.update_traces(textposition='inside', textinfo='percent+label', showlegend=False) # Hide legend for pie
                    st.plotly_chart(fig_err_label, use_container_width=True)
                    get_image_download_link(fig_err_label, "errors_by_label.png", "Download Errors by Label Chart (PNG)")
                else:
                     st.info("No errors to display in Errors by Label chart.")
            except Exception as e:
                st.error(f"Could not generate Errors by Label chart: {e}")
                logging.exception("Error generating Errors by Label chart")


        with col2:
            # Errors by Type (Response Code Category) (Bar Chart) - MODIFIED
            st.markdown("#### Errors by Response Code Category")
            try:
                if not errors_df.empty:
                    # Apply categorization
                    # Ensure responseCode is string before applying categorization
                    errors_df['responseCode'] = errors_df['responseCode'].astype(str).fillna("Unknown")
                    errors_df['Code Category'] = errors_df['responseCode'].apply(get_code_category)
                    # Group by category
                    errors_by_category = errors_df['Code Category'].value_counts().reset_index()
                    errors_by_category.columns = ['Code Category', 'Count']
                    errors_by_category = errors_by_category.sort_values(by='Count', ascending=False) # Sort by count

                    fig_err_cat = px.bar(errors_by_category, x='Code Category', y='Count',
                                          title='Error Count by Response Code Category',
                                          text='Count',
                                          # Define order for better visualization
                                          category_orders={"Code Category": ["2xx Success", "3xx Redirection", "4xx Client Error", "5xx Server Error", "Non HTTP", "Other", "Unknown"]}
                                          )
                    fig_err_cat.update_traces(textposition='outside')
                    fig_err_cat.update_layout(xaxis_title='Response Code Category', yaxis_title='Number of Errors')
                    st.plotly_chart(fig_err_cat, use_container_width=True)
                    get_image_download_link(fig_err_cat, "errors_by_category.png", "Download Errors by Category Chart (PNG)")
                else:
                     st.info("No errors to display in Errors by Code chart.")
            except Exception as e:
                st.error(f"Could not generate Errors by Code Category chart: {e}")
                logging.exception("Error generating Errors by Code Category chart")


        # --- Assertion Failures (Optional) ---
        if 'failureMessage' in df.columns:
             # Check for non-empty failure messages only on rows that are marked as failed (success=False)
             assertion_failures = df[df['failureMessage'].notna() & (df['failureMessage'] != '') & ~df['success']].copy()
             if not assertion_failures.empty:
                  st.markdown("#### Assertion Failures")
                  assertion_summary = assertion_failures.groupby(['label', 'failureMessage']).size().reset_index(name='Count')
                  assertion_summary = assertion_summary.sort_values('Count', ascending=False)
                  # Add URL if available
                  if 'URL' in df.columns:
                       try:
                            url_map_assert = assertion_failures.groupby(['label', 'failureMessage'])['URL'].first().reset_index()
                            assertion_summary = pd.merge(assertion_summary, url_map_assert, on=['label', 'failureMessage'], how='left')
                            assertion_summary['URL'] = assertion_summary['URL'].astype(str).str.slice(0, 100) + '...' # Truncate
                       except Exception as merge_err:
                            logging.warning(f"Could not merge URL into assertion failures: {merge_err}")

                  st.dataframe(assertion_summary, use_container_width=True)
                  get_table_download_link(assertion_summary, "assertion_failures.csv", "Download Assertion Failures (CSV)")


        # --- Comparison Logic (Simple Count Comparison) ---
        if comparison_df is not None and not comparison_df.empty: # Check .empty
            comp_errors_df = comparison_df[~comparison_df['success']].copy()
            comp_total_errors = len(comp_errors_df)
            comp_total_samples = len(comparison_df)
            comp_error_rate = (comp_total_errors / comp_total_samples) * 100 if comp_total_samples > 0 else 0

            st.markdown("#### Error Comparison with Baseline")
            col1_comp, col2_comp = st.columns(2)
            with col1_comp:
                 st.metric("Total Errors (Baseline)", f"{comp_total_errors:,} ({comp_error_rate:.2f}%)")
            with col2_comp:
                 delta = total_errors - comp_total_errors
                 delta_rate = error_rate - comp_error_rate
                 st.metric("Change in Errors", f"{delta:+,}", f"{delta_rate:+.2f}% points", delta_color="inverse") # Lower is better

        st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred during error analysis display: {e}")
        logging.exception("Error in display_error_analysis")


# --- NEW: Latency and Connect Time Charts ---
def display_latency_connect_charts(df, comparison_df=None):
    """Display Latency and Connect Time analysis charts."""
    st.markdown("<h2 class='sub-header'>Network Timing Analysis (Latency & Connect)</h2>", unsafe_allow_html=True)

    # Check if required columns exist
    latency_col_exists = 'Latency' in df.columns and pd.api.types.is_numeric_dtype(df['Latency'])
    connect_col_exists = 'Connect' in df.columns and pd.api.types.is_numeric_dtype(df['Connect'])

    if not latency_col_exists and not connect_col_exists:
        st.warning("Latency and Connect time columns not found or not numeric in the data. Cannot display Network Timing charts.")
        return
    if df is None or df.empty:
        st.warning("No data available for Network Timing analysis.")
        return

    tabs = []
    tab_names = []
    if latency_col_exists:
        tabs.append(None)
        tab_names.append("📈 Latency Over Time")
    if connect_col_exists:
        tabs.append(None)
        tab_names.append("🔗 Connect Time Over Time")

    if not tabs: # Should not happen based on initial check, but safety first
        return

    # Create tabs dynamically
    created_tabs = st.tabs(tab_names)
    tab_idx = 0

    # --- Latency Tab ---
    if latency_col_exists:
        with created_tabs[tab_idx]:
            st.markdown("#### Latency (Time to First Byte) Over Time")
            try:
                df_time = df.copy()
                df_time.set_index('timeStamp', inplace=True)

                # Determine resampling rate (same logic as response time)
                test_duration_sec = (df['timeStamp'].max() - df['timeStamp'].min()).total_seconds()
                resample_rate = '1min' if test_duration_sec > 3600 else '15s' if test_duration_sec > 600 else '5s'
                resample_label = '1 minute' if test_duration_sec > 3600 else '15 seconds' if test_duration_sec > 600 else '5 seconds'

                # Resample Latency data
                df_resampled = df_time.resample(resample_rate).agg({'Latency': ['mean', 'median', 'max']}).dropna()
                df_resampled.columns = ['Mean Latency', 'Median Latency', 'Max Latency']
                df_resampled = df_resampled.reset_index()

                fig_lat = go.Figure()
                if not df_resampled.empty:
                    fig_lat.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Mean Latency'], mode='lines', name='Mean Latency (Current)', line=dict(color='darkcyan')))
                    fig_lat.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Median Latency'], mode='lines', name='Median Latency (Current)', line=dict(color='lightseagreen')))
                    fig_lat.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Max Latency'], mode='lines', name='Max Latency (Current)', line=dict(color='red', dash='dot')))
                else:
                     st.warning("No data points available for Latency Trend after resampling.")

                # Add comparison if available
                if comparison_df is not None and not comparison_df.empty and 'Latency' in comparison_df.columns and pd.api.types.is_numeric_dtype(comparison_df['Latency']):
                    comp_df_time = comparison_df.copy()
                    comp_df_time.set_index('timeStamp', inplace=True)
                    comp_df_resampled = comp_df_time.resample(resample_rate).agg({'Latency': ['mean', 'median']}).dropna()
                    comp_df_resampled.columns = ['Mean Latency', 'Median Latency']
                    comp_df_resampled = comp_df_resampled.reset_index()
                    if not comp_df_resampled.empty:
                         fig_lat.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Mean Latency'], mode='lines', name='Mean Latency (Baseline)', line=dict(color='paleturquoise', dash='dash')))
                         fig_lat.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Median Latency'], mode='lines', name='Median Latency (Baseline)', line=dict(color='mediumaquamarine', dash='dash')))


                fig_lat.update_layout(title=f'Latency Over Time (Aggregated every {resample_label})', xaxis_title='Time', yaxis_title='Latency (ms)', height=400, hovermode="x unified")
                st.plotly_chart(fig_lat, use_container_width=True)
                get_image_download_link(fig_lat, "latency_trend.png", "Download Latency Trend (PNG)")

            except Exception as e:
                st.error(f"Could not generate Latency Trend chart: {e}")
                logging.exception("Error generating Latency Trend chart")
        tab_idx += 1


    # --- Connect Time Tab ---
    if connect_col_exists:
        with created_tabs[tab_idx]:
            st.markdown("#### Connect Time Over Time")
            try:
                df_time = df.copy()
                df_time.set_index('timeStamp', inplace=True)

                # Determine resampling rate
                test_duration_sec = (df['timeStamp'].max() - df['timeStamp'].min()).total_seconds()
                resample_rate = '1min' if test_duration_sec > 3600 else '15s' if test_duration_sec > 600 else '5s'
                resample_label = '1 minute' if test_duration_sec > 3600 else '15 seconds' if test_duration_sec > 600 else '5 seconds'

                # Resample Connect data
                df_resampled = df_time.resample(resample_rate).agg({'Connect': ['mean', 'median', 'max']}).dropna()
                df_resampled.columns = ['Mean Connect', 'Median Connect', 'Max Connect']
                df_resampled = df_resampled.reset_index()

                fig_conn = go.Figure()
                if not df_resampled.empty:
                    fig_conn.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Mean Connect'], mode='lines', name='Mean Connect (Current)', line=dict(color='indigo')))
                    fig_conn.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Median Connect'], mode='lines', name='Median Connect (Current)', line=dict(color='mediumpurple')))
                    fig_conn.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled['Max Connect'], mode='lines', name='Max Connect (Current)', line=dict(color='red', dash='dot')))
                else:
                     st.warning("No data points available for Connect Time Trend after resampling.")

                # Add comparison if available
                if comparison_df is not None and not comparison_df.empty and 'Connect' in comparison_df.columns and pd.api.types.is_numeric_dtype(comparison_df['Connect']):
                    comp_df_time = comparison_df.copy()
                    comp_df_time.set_index('timeStamp', inplace=True)
                    comp_df_resampled = comp_df_time.resample(resample_rate).agg({'Connect': ['mean', 'median']}).dropna()
                    comp_df_resampled.columns = ['Mean Connect', 'Median Connect']
                    comp_df_resampled = comp_df_resampled.reset_index()
                    if not comp_df_resampled.empty:
                         fig_conn.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Mean Connect'], mode='lines', name='Mean Connect (Baseline)', line=dict(color='thistle', dash='dash')))
                         fig_conn.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled['Median Connect'], mode='lines', name='Median Connect (Baseline)', line=dict(color='plum', dash='dash')))

                fig_conn.update_layout(title=f'Connect Time Over Time (Aggregated every {resample_label})', xaxis_title='Time', yaxis_title='Connect Time (ms)', height=400, hovermode="x unified")
                st.plotly_chart(fig_conn, use_container_width=True)
                get_image_download_link(fig_conn, "connect_time_trend.png", "Download Connect Time Trend (PNG)")

            except Exception as e:
                st.error(f"Could not generate Connect Time Trend chart: {e}")
                logging.exception("Error generating Connect Time Trend chart")
        tab_idx += 1

    st.markdown("---")

# --- NEW: Concurrency Charts ---
def display_concurrency_charts(df, comparison_df=None):
    """Display Active Threads over time if data is available."""
    st.markdown("<h2 class='sub-header'>Concurrency Analysis</h2>", unsafe_allow_html=True)

    # Check if required columns exist
    all_threads_col = 'allThreads' # Standard JMeter column name
    grp_threads_col = 'grpThreads'
    threads_col_exists = all_threads_col in df.columns and pd.api.types.is_numeric_dtype(df[all_threads_col])

    if not threads_col_exists:
        st.warning(f"'{all_threads_col}' column not found or not numeric. Cannot display Concurrency chart.")
        return
    if df is None or df.empty:
        st.warning("No data available for Concurrency analysis.")
        return

    st.markdown("#### Active Threads Over Time")
    try:
        # Drop rows where thread counts are NaN before resampling
        df_threads = df.dropna(subset=[all_threads_col]).copy()
        if grp_threads_col in df_threads.columns and pd.api.types.is_numeric_dtype(df_threads[grp_threads_col]):
             df_threads.dropna(subset=[grp_threads_col], inplace=True)

        if df_threads.empty:
             st.warning("No valid thread count data available for Concurrency chart.")
             return

        df_time = df_threads.set_index('timeStamp')


        # Determine resampling rate (use a slightly longer rate for threads maybe)
        test_duration_sec = (df_threads['timeStamp'].max() - df_threads['timeStamp'].min()).total_seconds()
        resample_rate = '1min' if test_duration_sec > 3600 else '30s' if test_duration_sec > 600 else '10s'
        resample_label = '1 minute' if test_duration_sec > 3600 else '30 seconds' if test_duration_sec > 600 else '10 seconds'

        # Resample thread data - use 'max' as threads usually step up/down
        agg_dict = {all_threads_col: 'max'}
        if grp_threads_col in df_threads.columns and pd.api.types.is_numeric_dtype(df_threads[grp_threads_col]):
             agg_dict[grp_threads_col] = 'max'

        df_resampled = df_time.resample(resample_rate).agg(agg_dict).dropna() # Drop intervals with no samples
        df_resampled = df_resampled.reset_index()

        fig_threads = go.Figure()

        if not df_resampled.empty:
            # Plot Total Active Threads
            fig_threads.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled[all_threads_col], mode='lines', name='Total Active Threads (Current)', line=dict(color='orangered', shape='hv'))) # Use step shape

            # Plot Group Threads if available
            if grp_threads_col in df_resampled.columns:
                 fig_threads.add_trace(go.Scatter(x=df_resampled['timeStamp'], y=df_resampled[grp_threads_col], mode='lines', name='Group Active Threads (Current)', line=dict(color='sandybrown', shape='hv', dash='dot')))
        else:
             st.warning("No data points available for Active Threads chart after resampling.")


        # Add comparison if available
        if (comparison_df is not None and not comparison_df.empty and
            all_threads_col in comparison_df.columns and pd.api.types.is_numeric_dtype(comparison_df[all_threads_col])):

            comp_df_threads = comparison_df.dropna(subset=[all_threads_col]).copy()
            if not comp_df_threads.empty:
                comp_df_time = comp_df_threads.set_index('timeStamp')
                comp_agg_dict = {all_threads_col: 'max'} # Only compare total threads for simplicity
                comp_df_resampled = comp_df_time.resample(resample_rate).agg(comp_agg_dict).dropna()
                comp_df_resampled = comp_df_resampled.reset_index()
                if not comp_df_resampled.empty:
                     fig_threads.add_trace(go.Scatter(x=comp_df_resampled['timeStamp'], y=comp_df_resampled[all_threads_col], mode='lines', name='Total Active Threads (Baseline)', line=dict(color='lightcoral', shape='hv', dash='dash')))


        fig_threads.update_layout(title=f'Active Threads Over Time (Sampled every {resample_label})', xaxis_title='Time', yaxis_title='Number of Active Threads', height=400, hovermode="x unified")
        st.plotly_chart(fig_threads, use_container_width=True)
        get_image_download_link(fig_threads, "active_threads_trend.png", "Download Active Threads Chart (PNG)")

    except Exception as e:
        st.error(f"Could not generate Active Threads chart: {e}")
        logging.exception("Error generating Active Threads chart")

    st.markdown("---")


# --- Main Application ---

def main():
    """Main function to run the Streamlit application."""
    st.markdown("<h1 class='main-header'>📊 JMeter Dashboard</h1>", unsafe_allow_html=True)
    st.write("Upload your JMeter test results (.jtl or .csv) to visualize performance metrics.")

    # --- Sidebar for File Upload and Options ---
    with st.sidebar:
        st.header("📁 Upload Files")
        uploaded_file = st.file_uploader("Upload Current Test Results (JTL/CSV)", type=['jtl', 'csv'], key="current")
        uploaded_baseline_file = st.file_uploader("Upload Baseline Test Results (Optional)", type=['jtl', 'csv'], key="baseline")

        st.header("⚙️ Options")
        # Add any options here if needed, e.g., delimiter override
        # delimiter_override = st.text_input("Delimiter (optional, e.g., '\\t' for tab)", "")
        # delimiter = delimiter_override if delimiter_override else None # Use override if provided

        # Placeholder for future options
        st.markdown("---")

    # --- Load Data ---
    # Initialize variables
    df_current = None
    df_baseline = None
    metrics_current = {}
    metrics_baseline = {}
    agg_report_current = None
    agg_report_baseline = None
    # all_labels = [] # No longer needed for multiselect
    min_ts_overall = None
    max_ts_overall = None

    # Load Current Data
    if uploaded_file:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            df_current = load_test_results(uploaded_file) # Pass delimiter if using override
            if df_current is not None and df_current.empty:
                 st.warning(f"File '{uploaded_file.name}' processed but resulted in an empty dataset. Check file content or processing logs.")
                 df_current = None # Treat as if loading failed if empty after processing
            elif df_current is None:
                 st.error(f"Failed to process '{uploaded_file.name}'. Check file format and content.")
                 # load_test_results already shows error messages
            elif df_current is not None:
                 # Get overall time range from current file if loaded successfully
                 min_ts_overall = df_current['timeStamp'].min()
                 max_ts_overall = df_current['timeStamp'].max()


    # Load Baseline Data
    if uploaded_baseline_file:
         with st.spinner(f"Processing baseline '{uploaded_baseline_file.name}'..."):
             df_baseline = load_test_results(uploaded_baseline_file) # Pass delimiter if using override
             if df_baseline is not None and df_baseline.empty:
                  st.warning(f"Baseline file '{uploaded_baseline_file.name}' processed but resulted in an empty dataset.")
                  df_baseline = None # Ignore empty baseline
             elif df_baseline is None:
                  st.warning(f"Failed to process baseline file '{uploaded_baseline_file.name}'. Comparison will not be available.")
                  # df_baseline is already None

    # --- Filters ---
    st.markdown("<h2 class='sub-header'>Filters</h2>", unsafe_allow_html=True)
    filter_col1, filter_col2 = st.columns([2,3]) # Allocate more space to time slider

    # -- Advanced Label Filter --
    label_filter_text = ""
    label_filter_type = "Contains" # Default filter type
    with filter_col1:
        st.markdown("**Filter by Request Label**") # Add sub-header for clarity
        filter_options = ["Contains", "Does Not Contain", "Equals", "Does Not Equal", "Begins with", "Does Not Begin with", "Ends with", "Does Not End with"]
        # Disable filter options if no data is loaded
        label_filter_disabled = (df_current is None or df_current.empty)

        label_filter_type = st.selectbox("Match Type:", options=filter_options, index=0, key="label_filter_type", disabled=label_filter_disabled)
        label_filter_text = st.text_input("Label Text:", key="label_filter_text", placeholder="Enter text to filter labels...", disabled=label_filter_disabled)


    # -- Time Slider Filter --
    selected_time_range = None
    with filter_col2:
        st.markdown("**Filter by Time Range**") # Add sub-header for clarity
        time_filter_disabled = True # Disabled by default
        if (
                min_ts_overall is not None and pd.notna(min_ts_overall) and
                max_ts_overall is not None and pd.notna(max_ts_overall) and
                min_ts_overall < max_ts_overall
        ):
             time_filter_disabled = False # Enable if valid range exists
             try:
                # Convert pandas timestamps to python datetime for slider compatibility if needed
                min_dt = min_ts_overall.to_pydatetime()
                max_dt = max_ts_overall.to_pydatetime()
                # Ensure value is within min/max bounds
                default_value = (min_dt, max_dt)

                selected_time_range_dt = st.slider(
                    "Time Window:",
                    min_value=min_dt,
                    max_value=max_dt,
                    value=default_value,
                    format="YYYY/MM/DD HH:mm:ss", # More standard format
                    step=timedelta(seconds=1), # Allow second precision
                    key="time_filter",
                    disabled=time_filter_disabled
                )
                # Convert back to pandas Timestamp for filtering
                selected_time_range = (pd.Timestamp(selected_time_range_dt[0]), pd.Timestamp(selected_time_range_dt[1]))

             except Exception as e:
                  st.error(f"Could not create time slider: {e}")
                  logging.error(f"Error creating time slider: min={min_ts_overall}, max={max_ts_overall}. Error: {e}")
                  selected_time_range = (min_ts_overall, max_ts_overall) # Fallback
        elif df_current is not None:
             st.info("Time range filter unavailable (invalid time range in data).")
        else:
             # Display placeholder slider if no data loaded
             st.slider("Time Window:", min_value=0, max_value=10, value=(0,10), disabled=True)


    # --- Filter DataFrames ---
    df_current_filtered = df_current
    df_baseline_filtered = df_baseline
    filter_applied_message = "Displaying data for all labels and full time range."
    label_filter_active = False
    time_filter_active = False

    # Apply Advanced Label Filter first
    if label_filter_text and not label_filter_disabled: # Only filter if text is entered and filter is enabled
        label_filter_active = True
        pattern = label_filter_text
        logging.info(f"Applying label filter: Type='{label_filter_type}', Pattern='{pattern}'")
        try:
            if df_current is not None:
                if label_filter_type == "Equals":             mask = df_current['label'] == pattern
                elif label_filter_type == "Does Not Equal":     mask = df_current['label'] != pattern
                elif label_filter_type == "Begins with":         mask = df_current['label'].str.startswith(pattern, na=False)
                elif label_filter_type == "Does Not Begin with": mask = ~df_current['label'].str.startswith(pattern, na=False)
                elif label_filter_type == "Ends with":           mask = df_current['label'].str.endswith(pattern, na=False)
                elif label_filter_type == "Does Not End with":   mask = ~df_current['label'].str.endswith(pattern, na=False)
                elif label_filter_type == "Contains":           mask = df_current['label'].str.contains(pattern, na=False, case=False, regex=False)
                elif label_filter_type == "Does Not Contain":   mask = ~df_current['label'].str.contains(pattern, na=False, case=False, regex=False)
                else: mask = pd.Series([True] * len(df_current)) # Should not happen

                df_current_filtered = df_current[mask]
                logging.info(f"Filtered current data to {len(df_current_filtered)} rows based on label filter.")
                if df_current_filtered.empty: st.warning("Current data is empty after applying label filter.")

            if df_baseline is not None:
                # Apply same filter logic to baseline
                if label_filter_type == "Equals":             mask_base = df_baseline['label'] == pattern
                elif label_filter_type == "Does Not Equal":     mask_base = df_baseline['label'] != pattern
                elif label_filter_type == "Begins with":         mask_base = df_baseline['label'].str.startswith(pattern, na=False)
                elif label_filter_type == "Does Not Begin with": mask_base = ~df_baseline['label'].str.startswith(pattern, na=False)
                elif label_filter_type == "Ends with":           mask_base = df_baseline['label'].str.endswith(pattern, na=False)
                elif label_filter_type == "Does Not End with":   mask_base = ~df_baseline['label'].str.endswith(pattern, na=False)
                elif label_filter_type == "Contains":           mask_base = df_baseline['label'].str.contains(pattern, na=False, case=False, regex=False)
                elif label_filter_type == "Does Not Contain":   mask_base = ~df_baseline['label'].str.contains(pattern, na=False, case=False, regex=False)
                else: mask_base = pd.Series([True] * len(df_baseline))

                df_baseline_filtered = df_baseline[mask_base]
                logging.info(f"Filtered baseline data to {len(df_baseline_filtered)} rows based on label filter.")

            filter_applied_message = f"Displaying data for labels where label {label_filter_type.lower()} '{pattern}'."

        except Exception as e:
             st.error(f"Error applying label filter: {e}")
             logging.exception("Error applying label filter")
             # Reset to unfiltered if error occurs
             df_current_filtered = df_current
             df_baseline_filtered = df_baseline
             label_filter_active = False
             filter_applied_message = "Error applying label filter. Displaying all data."


    # Apply Time Filter second (to potentially already label-filtered data)
    if selected_time_range and not time_filter_disabled and min_ts_overall is not pd.NaT and max_ts_overall is not pd.NaT:
         start_time_filter, end_time_filter = selected_time_range
         # Check if the slider has actually been moved from the full range
         # Add tolerance for timestamp comparison
         if start_time_filter > min_ts_overall + timedelta(seconds=1) or end_time_filter < max_ts_overall - timedelta(seconds=1):
              time_filter_active = True
              if df_current_filtered is not None:
                   # Apply time filter
                   df_current_filtered = df_current_filtered[
                       (df_current_filtered['timeStamp'] >= start_time_filter) &
                       (df_current_filtered['timeStamp'] <= end_time_filter)
                   ]
                   logging.info(f"Filtered current data to {len(df_current_filtered)} rows based on time range.")
                   if df_current_filtered.empty:
                        st.warning("Current data is empty after applying time filter.")

              if df_baseline_filtered is not None:
                   # Also filter baseline if slider is used
                   df_baseline_filtered = df_baseline_filtered[
                       (df_baseline_filtered['timeStamp'] >= start_time_filter) &
                       (df_baseline_filtered['timeStamp'] <= end_time_filter)
                   ]
                   logging.info(f"Filtered baseline data to {len(df_baseline_filtered)} rows based on time range.")

              # Update filter message
              time_msg = f" between {start_time_filter.strftime('%H:%M:%S')} and {end_time_filter.strftime('%H:%M:%S')}"
              if label_filter_active:
                   filter_applied_message = f"Displaying data for labels where label {label_filter_type.lower()} '{pattern}'{time_msg}."
              else:
                   filter_applied_message = f"Displaying data for all labels{time_msg}."


    st.caption(filter_applied_message)
    st.markdown("---") # Separator after filters


    # --- Calculate Metrics and Reports on Filtered Data ---
    # Recalculate metrics based on the final filtered data
    metrics_current = {}
    agg_report_current = None
    metrics_baseline = {}
    agg_report_baseline = None

    if df_current_filtered is not None and not df_current_filtered.empty:
        with st.spinner("Calculating metrics for selection..."):
             metrics_current = calculate_summary_metrics(df_current_filtered)
             agg_report_current = create_aggregate_report(df_current_filtered)
    else:
         if df_current is not None: # Only show warning if original data existed
             st.warning("No data matches the selected filter(s). Cannot display metrics or reports.")


    if df_baseline_filtered is not None and not df_baseline_filtered.empty:
         with st.spinner("Calculating baseline metrics for selection..."):
              metrics_baseline = calculate_summary_metrics(df_baseline_filtered)
              agg_report_baseline = create_aggregate_report(df_baseline_filtered)


    # --- Display Dashboard Sections ---
    # Display only if there's data AFTER filtering
    if df_current_filtered is not None and not df_current_filtered.empty:

        # Display Summary Metrics (with comparison if baseline exists)
        display_summary_metrics(metrics_current, metrics_baseline if df_baseline_filtered is not None else None)

        # Display Aggregate Report (with comparison if baseline exists)
        display_aggregate_report(agg_report_current, agg_report_baseline if df_baseline_filtered is not None else None)

        # Display Response Time Charts (with comparison if baseline exists)
        display_response_time_charts(df_current_filtered, df_baseline_filtered)

        # --- NEW: Display Latency/Connect Charts ---
        display_latency_connect_charts(df_current_filtered, df_baseline_filtered)

        # Display Throughput Charts (with comparison if baseline exists)
        display_throughput_charts(df_current_filtered, df_baseline_filtered)

        # --- NEW: Display Concurrency Chart ---
        display_concurrency_charts(df_current_filtered, df_baseline_filtered)

        # Display Error Analysis (with comparison if baseline exists)
        display_error_analysis(df_current_filtered, df_baseline_filtered)

        # --- Export Section ---
        st.markdown("<h2 class='sub-header'>Export Options</h2>", unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2) # Create columns for buttons

        # PDF Export
        with export_col1:
            if PDF_EXPORTER_AVAILABLE:
                st.caption("Generate a PDF summary of the current view.")
                if st.button("Generate PDF Report", key="pdf_button"):
                    # Check again if data exists before generating PDF
                    if df_current_filtered is None or df_current_filtered.empty:
                         st.error("Cannot generate PDF report: No data available for the current selection.")
                    else:
                        with st.spinner("Generating PDF report... This may take a moment."):
                            try:
                                # Create temporary directory for images
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    logging.info(f"Using temporary directory for PDF images: {temp_dir}")
                                    # Prepare report data using FINAL FILTERED dataframes and metrics
                                    report_data = {
                                        "summary_metrics": metrics_current, # Use recalculated metrics
                                        "comparison_summary_metrics": metrics_baseline if df_baseline_filtered is not None else None, # Use recalculated baseline metrics
                                        "aggregate_report": agg_report_current, # Use recalculated aggregate report
                                        "comparison_aggregate_report": agg_report_baseline if df_baseline_filtered is not None else None,
                                        # Ensure error breakdown is generated *using final filtered data*
                                        "error_breakdown": create_error_breakdown_table(df_current_filtered),
                                        "df_current": df_current_filtered, # Pass FINAL FILTERED dataframes
                                        "df_baseline": df_baseline_filtered
                                        # Note: PDF exporter itself needs updates to show new metrics/charts
                                    }

                                    # Generate the PDF using the imported function
                                    pdf_bytes_io = generate_report(report_data, temp_dir) # Expecting BytesIO object

                                    if pdf_bytes_io:
                                        st.success("PDF report generated successfully!")
                                        st.download_button(
                                            label="Download PDF Report",
                                            data=pdf_bytes_io, # Pass BytesIO directly
                                            file_name=f"jmeter_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf"
                                        )
                                    else:
                                        st.error("Failed to generate PDF report. Check logs for details.")

                            except Exception as e:
                                st.error(f"An error occurred during PDF generation: {e}")
                                logging.exception("Error during PDF generation button click") # Log full traceback
            else:
                st.warning("PDF export is disabled ('pdf_exporter.py' missing).")

        # --- NEW: Email Generation ---
        with export_col2:
            st.caption("Generate an email draft with key metrics.")
            # Prepare email content only if metrics are available
            if metrics_current:
                try:
                    # Select key metrics for email body
                    email_subject = f"JMeter Test Results Summary ({uploaded_file.name if uploaded_file else 'N/A'})"
                    email_body_lines = [
                        f"JMeter Test Results Summary:",
                        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Source File: {uploaded_file.name if uploaded_file else 'N/A'}",
                        f"Applied Filters: {filter_applied_message}", # Include filter status
                        "--- Key Metrics ---",
                        f"Duration: {metrics_current.get('Formatted Duration', 'N/A')}",
                        f"Total Samples: {metrics_current.get('Total Samples', 'N/A'):,}",
                        f"Avg Response Time: {metrics_current.get('Avg Response Time', 'N/A'):.2f} ms" if isinstance(metrics_current.get('Avg Response Time'), (int, float)) else f"Avg Response Time: N/A",
                        f"95th Percentile: {metrics_current.get('95th Percentile', 'N/A'):.2f} ms" if isinstance(metrics_current.get('95th Percentile'), (int, float)) else f"95th Percentile: N/A",
                        f"Throughput: {metrics_current.get('Throughput', 'N/A'):.2f} req/sec" if isinstance(metrics_current.get('Throughput'), (int, float)) else f"Throughput: N/A",
                        f"Error Rate: {metrics_current.get('Error Rate', 'N/A'):.2f}%" if isinstance(metrics_current.get('Error Rate'), (int, float)) else f"Error Rate: N/A",
                        f"Avg Latency: {metrics_current.get('Avg Latency', 'N/A'):.2f} ms" if isinstance(metrics_current.get('Avg Latency'), (int, float)) else f"Avg Latency: N/A",
                        f"Avg Bandwidth: {format_bandwidth(metrics_current.get('Avg Bandwidth (Recv)', 'N/A'))}",
                        "---",
                        "Note: This is a summary based on the current dashboard view.",
                    ]
                    email_body = "\n".join(email_body_lines)

                    # URL Encode
                    encoded_subject = urllib.parse.quote(email_subject)
                    encoded_body = urllib.parse.quote(email_body)

                    # Create mailto link
                    mailto_link = f"mailto:?subject={encoded_subject}&body={encoded_body}"

                    # Create HTML button link
                    button_html = f"""
                    <a href="{mailto_link}" target="_blank">
                        <button style="width: 100%; background-color: #1E88E5; color: white; border-radius: 5px; padding: 0.5rem 1rem; border: none; cursor: pointer; transition: background-color 0.3s ease;">
                            Generate Email Draft
                        </button>
                    </a>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Could not generate email link: {e}")
                    logging.exception("Error generating mailto link")
            else:
                # Disable button if no metrics
                st.button("Generate Email Draft", key="email_button", disabled=True)


    # Handle case where initial file upload failed or was empty
    elif df_current is None and uploaded_file:
         # Error message already shown by load_test_results
         pass # Do nothing more, wait for valid upload
    # elif not all_labels and uploaded_file: # all_labels is no longer used for filter check
    #      # File loaded but no labels found (maybe empty file or only header?)
    #      st.warning("Could not extract any request labels from the uploaded file.")


if __name__ == "__main__":
    # Add a basic check for pdf_exporter availability at the start
    if not PDF_EXPORTER_AVAILABLE:
         # Display warning in the main area if sidebar isn't guaranteed yet
         st.warning("PDF Export Disabled:\n'pdf_exporter.py' not found or failed to load.", icon="⚠️")
    main()

# Helper function create_error_breakdown_table (moved to end for readability)
def create_error_breakdown_table(df):
    """
    Create a DataFrame summarizing errors by label, response code, and message.
    Returns a DataFrame, potentially empty, or a DataFrame with an 'Error' column on failure.
    """
    if df is None or df.empty or 'success' not in df.columns:
        logging.warning("Cannot create error breakdown: DataFrame is None, empty, or missing 'success' column.")
        # Return empty DF, not the error DF, as the condition is known upfront
        return pd.DataFrame()

    try:
        error_df = df[~df['success']].copy()

        if len(error_df) == 0:
            logging.info("No errors found in the data for breakdown.")
            return pd.DataFrame() # Return empty DataFrame if no errors

        group_by_cols = ['label', 'responseCode']
        # Add responseMessage to grouping if it exists
        if 'responseMessage' in error_df.columns:
             # Truncate long response messages for better table display
             error_df['responseMessage'] = error_df['responseMessage'].astype(str).str.slice(0, 70) # Limit message length
             group_by_cols.append('responseMessage')
        else:
             logging.warning("Column 'responseMessage' not found for error breakdown grouping.")

        # Add URL to grouping if it exists
        if 'URL' in error_df.columns:
             # Truncate long URLs
             error_df['URL'] = error_df['URL'].astype(str).str.slice(0, 100) + '...'
             group_by_cols.append('URL')
        else:
             logging.warning("Column 'URL' not found for error breakdown grouping.")


        # Ensure columns exist before grouping
        missing_group_cols = [col for col in group_by_cols if col not in error_df.columns]
        if missing_group_cols:
             logging.error(f"Missing columns required for error grouping: {missing_group_cols}")
             return pd.DataFrame({'Error': [f"Missing required columns: {', '.join(missing_group_cols)}"]})


        # Group and summarize
        if pd.__version__ >= '2.1.0':
             error_summary = error_df.groupby(group_by_cols, observed=False).size().reset_index(name='Count')
        else:
             error_summary = error_df.groupby(group_by_cols).size().reset_index(name='Count')

        error_summary = error_summary.sort_values('Count', ascending=False)

        # Calculate Percentage
        total_errors = error_summary['Count'].sum()
        if total_errors > 0:
             error_summary['Percentage'] = (error_summary['Count'] / total_errors * 100).apply(lambda x: f"{x:.2f}%")
        else:
             error_summary['Percentage'] = "0.00%" # Should not happen if error_df wasn't empty, but safe fallback


        return error_summary

    except Exception as e:
        logging.exception(f"Error grouping errors for breakdown table: {e}")
        # Return the specific fallback DataFrame with 'Error' column
        return pd.DataFrame({'Error': [f"Failed to generate breakdown: {e}"]})
