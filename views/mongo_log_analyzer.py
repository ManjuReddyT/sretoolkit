import streamlit as st
import pandas as pd
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Get the absolute path to the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to Python path if not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import the MongoDB parser functions
try:
    from scripts.mongoLogParser import process_log_lines, write_output_file, generate_summary_txt
except ImportError:
    st.error("""
    Failed to import MongoDB parser functions. 
    Please make sure you're running the Streamlit app from the root directory:
    ```
    streamlit run streamlit_app.py
    ```
    """)

def display_dataframe_with_filters(df, title, key_prefix):
    st.subheader(title)
    
    if not df.empty:
        # Convert any dictionary columns to strings
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].apply(lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else x)
        
        # Create column configurations
        column_config = {}
        for col in display_df.columns:
            if col in ['Command', 'Filter', 'Sample Commands']:
                column_config[col] = st.column_config.TextColumn(
                    help="Click to expand/copy",
                    width="large",
                )
            elif display_df[col].dtype in ['float64', 'int64']:
                if 'Duration' in col:
                    column_config[col] = st.column_config.NumberColumn(
                        help="Duration in milliseconds",
                        format="%.2f ms"
                    )
                else:
                    column_config[col] = st.column_config.NumberColumn(
                        format="%d"
                    )
        
        # Display the interactive dataframe
        st.data_editor(
            display_df,
            column_config=column_config,
            hide_index=True,
            num_rows="dynamic",
            key=f"{key_prefix}_editor"
        )
        
        # Add download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {title} as CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}.csv",
            mime='text/csv',
            key=f"{key_prefix}_download"
        )

def app():
    st.title("MongoDB Log Analyzer")
    st.write("Upload your MongoDB log file and analyze its contents")

    # File uploader
    uploaded_file = st.file_uploader("Choose a MongoDB log file", type=['log', 'txt', 'json'])
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date (Optional)",
            value=None,
            help="Filter logs after this date"
        )
        start_time = st.time_input(
            "Start Time",
            value=datetime.strptime("00:00:00", "%H:%M:%S").time(),
            help="Filter logs after this time"
        )
    with col2:
        end_date = st.date_input(
            "End Date (Optional)",
            value=None,
            help="Filter logs before this date"
        )
        end_time = st.time_input(
            "End Time",
            value=datetime.strptime("23:59:59", "%H:%M:%S").time(),
            help="Filter logs before this time"
        )
    
    # Combine date and time if date is selected
    start_datetime = None
    end_datetime = None
    if start_date:
        start_datetime = datetime.combine(start_date, start_time)
    if end_date:
        end_datetime = datetime.combine(end_date, end_time)

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Read the file
            with open(tmp_file_path, 'r') as file:
                lines = file.readlines()
            total_lines_read = len(lines)

            # Process the log lines
            output_df, non_slow_query_df, error_df, query_stats_df, remote_ip_df, error_type_df, operation_stats_df, total_operations = process_log_lines(
                lines, start_datetime, end_datetime
            )

            # Display summary statistics
            st.header("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Lines", total_lines_read)
            with col2:
                st.metric("Slow Queries", len(output_df))
            with col3:
                st.metric("Errors", len(error_df))
            with col4:
                st.metric("Unique IPs", len(remote_ip_df))

            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Slow Queries", 
                "Query Stats", 
                "Remote IP Stats", 
                "Error Details",
                "Error Types",
                "Non-Slow Queries",
                "Operation Stats"
            ])

            with tab1:
                display_dataframe_with_filters(output_df, "Detailed Slow Queries", "slow")

            with tab2:
                display_dataframe_with_filters(query_stats_df, "Normalized Query Statistics", "query")

            with tab3:
                display_dataframe_with_filters(remote_ip_df, "Remote IP Statistics", "ip")

            with tab4:
                display_dataframe_with_filters(error_df, "Detailed Errors", "error")

            with tab5:
                display_dataframe_with_filters(error_type_df, "Error Type Statistics", "error_type")

            with tab6:
                display_dataframe_with_filters(non_slow_query_df, "Non-Slow Queries", "nonslow")

            with tab7:
                display_dataframe_with_filters(operation_stats_df, "Operation Statistics", "op")

            # Generate Excel report
            if st.button("Generate Excel Report"):
                # Create temporary file for Excel output
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                    excel_path = tmp_excel.name
                    
                # Write Excel file
                write_output_file(
                    excel_path, output_df, non_slow_query_df, error_df,
                    query_stats_df, remote_ip_df, error_type_df, operation_stats_df
                )
                
                # Generate summary
                generate_summary_txt(
                    excel_path, total_lines_read, total_operations,
                    output_df, query_stats_df, remote_ip_df, error_df,
                    error_type_df, operation_stats_df
                )
                
                # Read the files for download
                with open(excel_path, 'rb') as excel_file:
                    excel_data = excel_file.read()
                with open(excel_path.replace('.xlsx', '_summary.txt'), 'rb') as summary_file:
                    summary_data = summary_file.read()
                
                # Create download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name="mongodb_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        label="Download Summary Report",
                        data=summary_data,
                        file_name="mongodb_analysis_summary.txt",
                        mime="text/plain"
                    )

                # Cleanup temporary files
                os.unlink(excel_path)
                os.unlink(excel_path.replace('.xlsx', '_summary.txt'))

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
        finally:
            # Cleanup temporary upload file
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    app()
