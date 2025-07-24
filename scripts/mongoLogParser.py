#!/usr/bin/env python3
import sys
import pandas as pd
import json
import os
import re
from collections import defaultdict
from statistics import mean

# Function to normalize queries
def normalize_query(query):
    """
    Normalizes a MongoDB query string by replacing specific values with a placeholder.
    This helps in grouping similar queries regardless of the exact values used in filters.
    It targets numbers, strings, booleans, null, and arrays of various types.
    """
    # Regex to replace values in key-value pairs (numbers, strings, booleans, nulls)
    # The lookahead ensures replacement only happens where a comma or closing brace/bracket follows.
    normalized_query = re.sub(r'(:\s*(?:null|true|false|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|["\'](?:[^"\\]|\\.)*?["\'])\s*(?=[,}\]]))', ':<value>', query)
    # Regex to replace array values (e.g., : [1, 2, 3] or : ["a", "b"])
    normalized_query = re.sub(r'(:\s*\[\s*(?:[^\]]*?)\s*\])', ':<value>', normalized_query)
    return normalized_query

def clean_json_string(json_str):
    """
    Cleans a JSON string by removing unbalanced single quotes at the beginning or end.
    This is useful for logs where JSON payloads might be improperly quoted.
    """
    if json_str.startswith("'"):
        json_str = json_str[1:]
    if json_str.endswith("'") and not json_str.endswith("\\'"): # Ensure it's not an escaped quote
        json_str = json_str[:-1]
    return json_str

def get_operation_type(command):
    """
    Identifies the type of MongoDB operation (e.g., 'find', 'insert', 'update')
    from the command dictionary.
    """
    # Common top-level command keys indicate the operation type
    for op_key in ['find', 'insert', 'update', 'delete', 'aggregate', 'count', 'distinct', 'mapReduce',
                   'createIndexes', 'dropIndexes', 'listCollections', 'listDatabases', 'replSetGetStatus',
                   'ismaster', 'getMore', 'hello', 'ping', 'saslStart', 'authenticate', 'setFeatureCompatibilityVersion']:
        if op_key in command:
            return op_key
    return 'other' # Fallback for commands not explicitly covered

def process_log_lines(lines, start_time, end_time):
    """
    Processes each log line, extracts relevant data, and aggregates statistics.
    Separates slow queries, non-slow queries, and errors.
    """
    output_columns = [
        'Command', 'Collection', 'DBName', 'APPName', 'Duration(ms)', 'KeysExamined',
        'DocsExamined', 'numYields', 'nreturned', 'Filter', 'Plan', 'RemoteIP',
        'timestamp', 'Operation Type'
    ]
    error_columns = ['Line_Num', 'msg', 'error_code_name', 'error_message', 'Raw_Line_Sample']

    data = [] # For detailed slow query metrics
    non_slow_query_data = [] # For raw non-slow query lines
    error_data = [] # For detailed error occurrences

    # Defaultdicts for aggregated statistics
    query_stats = defaultdict(lambda: {
        "count": 0,
        "durations": [],
        "keys_examined": [],
        "docs_examined": [],
        "num_yields": [],
        "nreturned": [],
        "sample_command": "",
        "sample_plan": ""
    })
    remote_ip_stats = defaultdict(lambda: {
        "count": 0,
        "durations": [],
        "collections_involved": set(),
        "sample_commands": set()
    })
    error_type_stats = defaultdict(lambda: {"totalCount": 0, "SampleLine": ""})

    # NEW: Operation Type Stats aggregation
    operation_stats = defaultdict(lambda: {
        "count": 0,
        "durations": [],
        "keys_examined": [],
        "docs_examined": [],
        "num_yields": [],
        "nreturned": [],
    })

    total_operations_processed_for_stats = 0 # This will count only operations that pass parsing and time filter

    for index, line in enumerate(lines):
        try:
            # Skip non-JSON lines or lines that don't look like MongoDB log entries
            if not line.strip().startswith('{') or ('"t":' not in line and '"ts":' not in line):
                non_slow_query_data.append({"Line_Num": index + 1, "Raw_Line": line.strip()})
                continue

            json_payload = json.loads(clean_json_string(line))

            # Extract timestamp for time filtering
            line_time_str = json_payload.get('t', {}).get('$date', '') or json_payload.get('ts', {}).get('$date', '')
            if not line_time_str:
                print(f"Warning: Line {index+1} has no recognizable timestamp. Skipping time filtering for this line.")
                line_time = None
            else:
                try:
                    line_time = pd.to_datetime(line_time_str)
                except ValueError:
                    print(f"Warning: Could not parse timestamp '{line_time_str}' in line {index+1}. Skipping time filtering for this line.")
                    line_time = None

            # Apply time filtering if start and end times are provided and line_time is valid
            if line_time and start_time and end_time and not (start_time <= line_time <= end_time):
                continue

            # Increment count for total operations that passed initial checks and time filter
            total_operations_processed_for_stats += 1

            attr = json_payload.get('attr', {})
            command = attr.get('command', {})
            ns_full = attr.get('ns', '')
            ns_parts = ns_full.split('.')
            db_name = ns_parts[0] if ns_parts else ''
            collection = ns_parts[1] if len(ns_parts) > 1 else ''
            app_name = attr.get('appName', '')
            duration = attr.get('durationMillis', 0)
            keys_examined = attr.get('keysExamined', 0)
            docs_examined = attr.get('docsExamined', 0)
            num_yields = attr.get('numYields', 0)
            nreturned = attr.get('nreturned', 0)
            
            full_remote_address = attr.get('remote', '')
            remote_ip = full_remote_address.split(':')[0] if ':' in full_remote_address else full_remote_address

            operation_type = get_operation_type(command)

            filter_ = {}
            if 'pipeline' in command:
                if command.get('pipeline') and isinstance(command['pipeline'], list) and command['pipeline']:
                    match_stage = next((stage.get('$match', {}) for stage in command['pipeline'] if isinstance(stage, dict) and '$match' in stage), {})
                    filter_ = match_stage
            elif 'filter' in command:
                filter_ = command.get('filter', {})
            
            plan = attr.get('planSummary', '')
            
            is_slow_query = "Slow query" in json_payload.get('msg', '') or duration > 100 # Example threshold: >100ms

            if is_slow_query:
                # Convert dictionary to JSON string for the command field
                command_str = json.dumps(command) if command else "{}"
                data.append([
                    command_str, collection, db_name, app_name, duration, keys_examined,
                    docs_examined, num_yields, nreturned, json.dumps(filter_), plan,
                    remote_ip, line_time_str, operation_type
                ])

                normalized_cmd_str = json.dumps(command)
                normalized_query = normalize_query(normalized_cmd_str)
                query_stats[normalized_query]["count"] += 1
                query_stats[normalized_query]["durations"].append(duration)
                query_stats[normalized_query]["keys_examined"].append(keys_examined)
                query_stats[normalized_query]["docs_examined"].append(docs_examined)
                query_stats[normalized_query]["num_yields"].append(num_yields)
                query_stats[normalized_query]["nreturned"].append(nreturned)
                if not query_stats[normalized_query]["sample_command"]:
                    query_stats[normalized_query]["sample_command"] = normalized_cmd_str
                if not query_stats[normalized_query]["sample_plan"]:
                    query_stats[normalized_query]["sample_plan"] = plan

                if remote_ip:
                    remote_ip_stats[remote_ip]["count"] += 1
                    remote_ip_stats[remote_ip]["durations"].append(duration)
                    if collection:
                        remote_ip_stats[remote_ip]["collections_involved"].add(collection)
                    if len(remote_ip_stats[remote_ip]["sample_commands"]) < 5:
                        remote_ip_stats[remote_ip]["sample_commands"].add(normalized_cmd_str)
            else:
                non_slow_query_data.append({"Line_Num": index + 1, "Raw_Line": line.strip()})

            # Populate Operation Type Statistics for ALL relevant operations
            op_key = (db_name, collection, operation_type)
            operation_stats[op_key]["count"] += 1
            operation_stats[op_key]["durations"].append(duration)
            operation_stats[op_key]["keys_examined"].append(keys_examined)
            operation_stats[op_key]["docs_examined"].append(docs_examined)
            operation_stats[op_key]["num_yields"].append(num_yields)
            operation_stats[op_key]["nreturned"].append(nreturned)


            if ('msg' in json_payload and json_payload.get('severity') == 'E') or ('error' in attr):
                msg = json_payload.get('msg', '')
                error_code_name = attr.get('error', {}).get('codeName', '')
                error_message = attr.get('error', {}).get('errmsg', '')

                error_type_stats[msg]["totalCount"] += 1
                if not error_type_stats[msg]["SampleLine"]:
                    error_type_stats[msg]["SampleLine"] = line.strip()
                
                error_data.append([index + 1, msg, error_code_name, error_message, line.strip()])

        except json.JSONDecodeError:
            error_data.append([index + 1, "JSONDecodeError", "", f"Invalid JSON: {line.strip()[:100]}...", line.strip()])
            print(f"Error: Invalid JSON payload in line {index + 1}. Skipping line.")
        except Exception as e:
            error_data.append([index + 1, "ProcessingError", "", f"Error: {e}, Line: {line.strip()[:100]}...", line.strip()])
            print(f"Error processing line {index + 1}: {e}. Skipping line.")

    # --- Create DataFrames from collected data ---
    output_df = pd.DataFrame(data, columns=output_columns)
    non_slow_query_df = pd.DataFrame(non_slow_query_data)
    error_df = pd.DataFrame(error_data, columns=error_columns)

    query_stats_list = []
    for q, v in query_stats.items():
        durations = v["durations"]
        keys_examined = v["keys_examined"]
        docs_examined = v["docs_examined"]
        num_yields = v["num_yields"]
        nreturned = v["nreturned"]

        query_stats_list.append({
            "Normalized Query": q,
            "Executions": v["count"],
            "Min Duration(ms)": min(durations) if durations else 0,
            "Max Duration(ms)": max(durations) if durations else 0,
            "Avg Duration(ms)": mean(durations) if durations else 0,
            "Avg Keys Examined": mean(keys_examined) if keys_examined else 0,
            "Avg Docs Examined": mean(docs_examined) if docs_examined else 0,
            "Avg numYields": mean(num_yields) if num_yields else 0,
            "Avg nreturned": mean(nreturned) if nreturned else 0,
            "Sample Command": v["sample_command"],
            "Sample Plan": v["sample_plan"]
        })
    query_stats_df = pd.DataFrame(query_stats_list)

    remote_ip_stats_list = []
    for ip, v in remote_ip_stats.items():
        durations = v["durations"]
        remote_ip_stats_list.append({
            "Remote IP": ip,
            "Executions": v["count"],
            "Min Duration(ms)": min(durations) if durations else 0,
            "Max Duration(ms)": max(durations) if durations else 0,
            "Avg Duration(ms)": mean(durations) if durations else 0,
            "Collections Involved": ", ".join(sorted(v["collections_involved"])),
            "Sample Commands": "\n".join(list(v["sample_commands"]))
        })
    remote_ip_df = pd.DataFrame(remote_ip_stats_list)

    error_type_stats_list = []
    for msg, v in error_type_stats.items():
        error_type_stats_list.append({
            "Error Message Type": msg,
            "Total Occurrences": v["totalCount"],
            "Sample Line": v["SampleLine"]
        })
    error_type_df = pd.DataFrame(error_type_stats_list)

    # NEW: Prepare Operation Type Stats DataFrame
    operation_stats_data = []
    for (db, collection, op_type), stats in operation_stats.items():
        durations = stats["durations"]
        keys_examined = stats["keys_examined"]
        docs_examined = stats["docs_examined"]
        num_yields = stats["num_yields"]
        nreturned = stats["nreturned"]

        percentage = (stats["count"] / total_operations_processed_for_stats * 100) if total_operations_processed_for_stats > 0 else 0

        operation_stats_data.append({
            "DBName": db,
            "Collection": collection,
            "Operation Type": op_type,
            "Executions": stats["count"],
            "Avg Duration(ms)": mean(durations) if durations else 0,
            "Avg Keys Examined": mean(keys_examined) if keys_examined else 0,
            "Avg Docs Examined": mean(docs_examined) if docs_examined else 0,
            "Avg numYields": mean(num_yields) if num_yields else 0,
            "Avg nreturned": mean(nreturned) if nreturned else 0,
            "Percentage of Total (%)": percentage
        })
    # Sort by DB, Collection, then Operation Type for better readability
    operation_stats_df = pd.DataFrame(operation_stats_data).sort_values(
        by=["DBName", "Collection", "Operation Type"]).reset_index(drop=True)

    return output_df, non_slow_query_df, error_df, query_stats_df, remote_ip_df, error_type_df, operation_stats_df, total_operations_processed_for_stats

def write_output_file(output_file_path, output_df, non_slow_query_df, error_df, query_stats_df, remote_ip_df, error_type_df, operation_stats_df):
    """
    Writes all generated DataFrames to an Excel file, each on a separate sheet.
    Applies autofilters and adjusts column widths for better readability.
    """
    try:
        with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
            # Write each DataFrame to a separate sheet
            output_df.to_excel(writer, sheet_name='Detailed Slow Queries', index=False)
            query_stats_df.to_excel(writer, sheet_name='Normalized Query Stats', index=False)
            remote_ip_df.to_excel(writer, sheet_name='Remote IP Stats', index=False)
            error_df.to_excel(writer, sheet_name='Detailed Errors', index=False)
            error_type_df.to_excel(writer, sheet_name='Error Type Stats', index=False)
            non_slow_query_df.to_excel(writer, sheet_name='Non-Slow Queries Raw', index=False)
            operation_stats_df.to_excel(writer, sheet_name='Operation Type Stats', index=False) # Fixed: Now correctly written

            # Apply formatting (autofilter and column widths) to each sheet
            workbook = writer.book
            sheets_to_format = {
                'Detailed Slow Queries': output_df,
                'Normalized Query Stats': query_stats_df,
                'Remote IP Stats': remote_ip_df,
                'Detailed Errors': error_df,
                'Error Type Stats': error_type_df,
                'Non-Slow Queries Raw': non_slow_query_df,
                'Operation Type Stats': operation_stats_df # Fixed: Included for formatting
            }

            for sheet_name, df_to_write in sheets_to_format.items():
                worksheet = writer.sheets[sheet_name]
                if not df_to_write.empty:
                    # Add autofilter to the header row
                    worksheet.autofilter(0, 0, len(df_to_write), len(df_to_write.columns) - 1)
                    # Adjust column widths dynamically
                    for col_num, column_name in enumerate(df_to_write.columns):
                        max_len = max(df_to_write[column_name].astype(str).map(len).max(), len(column_name))
                        worksheet.set_column(col_num, col_num, min(max_len + 2, 80)) # Limit max width to 80 characters

        print(f"Output successfully written to '{output_file_path}'")
    except Exception as e:
        print(f"Error writing the output file: {e}")
        sys.exit(1)

def generate_summary_txt(output_excel_path, total_lines_read, total_operations_processed,
                         output_df, query_stats_df, remote_ip_df, error_df, error_type_df, operation_stats_df, top_n=10):
    """
    Generates a comprehensive summary text file of the MongoDB log analysis.
    """
    summary_txt_path = os.path.splitext(output_excel_path)[0] + "_summary.txt"

    with open(summary_txt_path, 'w') as f:
        f.write("📊 MongoDB Log Analysis Summary\n")
        f.write("===================================\n\n")

        # Infer input file name if possible, or use a placeholder
        input_file_name_display = "N/A"
        # Attempt to derive the original input filename if the output excel name follows a pattern
        # This is a heuristic and might not always work perfectly.
        base_output_name = os.path.basename(output_excel_path)
        if base_output_name.endswith('.xlsx'):
            potential_input_name = base_output_name.replace('.xlsx', '.csv')
            if os.path.exists(os.path.join(os.path.dirname(output_excel_path), potential_input_name)):
                input_file_name_display = potential_input_name
            else:
                input_file_name_display = "Source unknown (check input JSON)"


        f.write(f"Input File: {input_file_name_display}\n")
        f.write(f"Output Excel File: {os.path.basename(output_excel_path)}\n")
        f.write(f"Total Log Lines Read: {total_lines_read}\n")
        f.write(f"Total Log Operations Processed (after time filter): {total_operations_processed}\n")
        f.write(f"Slow Queries Identified: {len(output_df)}\n")
        f.write(f"Unique Normalized Queries: {len(query_stats_df)}\n")
        f.write(f"Unique Remote IPs: {len(remote_ip_df)}\n")
        f.write(f"Total Error Occurrences: {len(error_df)}\n")
        f.write(f"Unique Error Types: {len(error_type_df)}\n\n")

        f.write(f"Top {top_n} Slowest Normalized Queries (by Avg Duration):\n")
        f.write("---------------------------------------------------\n")
        if not query_stats_df.empty:
            sorted_queries = query_stats_df.sort_values(by='Avg Duration(ms)', ascending=False)
            for idx, row in sorted_queries.head(top_n).iterrows():
                f.write(f"  Query: {row['Normalized Query']}\n")
                f.write(f"    Executions: {row['Executions']}\n")
                f.write(f"    Avg Duration: {row['Avg Duration(ms)']:.2f} ms\n")
                f.write(f"    Avg Keys Examined: {row['Avg Keys Examined']:.2f}\n")
                f.write(f"    Avg Docs Examined: {row['Avg Docs Examined']:.2f}\n")
                f.write(f"    Avg numYields: {row['Avg numYields']:.2f}\n")
                f.write(f"    Avg nreturned: {row['Avg nreturned']:.2f}\n")
                f.write(f"    Sample Command: {row['Sample Command'][:100]}...\n")
                f.write("\n")
        else:
            f.write("  No slow query statistics available.\n\n")

        f.write(f"Top {top_n} Remote IPs (by Total Executions):\n")
        f.write("------------------------------------------\n")
        if not remote_ip_df.empty:
            sorted_ips = remote_ip_df.sort_values(by='Executions', ascending=False)
            for idx, row in sorted_ips.head(top_n).iterrows():
                f.write(f"  Remote IP: {row['Remote IP']}\n")
                f.write(f"    Executions: {row['Executions']}\n")
                f.write(f"    Avg Duration: {row['Avg Duration(ms)']:.2f} ms\n")
                f.write(f"    Collections Involved: {row['Collections Involved']}\n")
                f.write("\n")
        else:
            f.write("  No remote IP statistics available.\n\n")

        f.write(f"Top {top_n} Error Message Types (by Occurrences):\n")
        f.write("----------------------------------------------\n")
        if not error_type_df.empty:
            sorted_errors = error_type_df.sort_values(by='Total Occurrences', ascending=False)
            for idx, row in sorted_errors.head(top_n).iterrows():
                f.write(f"  Error Type: {row['Error Message Type']}\n")
                f.write(f"    Total Occurrences: {row['Total Occurrences']}\n")
                f.write(f"    Sample Line: {row['Sample Line'][:100]}...\n")
                f.write("\n")
        else:
            f.write("  No error statistics available.\n\n")

        # NEW: Summary for Operation Type Statistics
        f.write(f"Top {top_n} Operation Types by Executions (DB, Collection, Type):\n")
        f.write("----------------------------------------------------------------\n")
        if not operation_stats_df.empty:
            # Sort by executions for overall top operations
            sorted_ops = operation_stats_df.sort_values(by='Executions', ascending=False)
            for idx, row in sorted_ops.head(top_n).iterrows():
                f.write(f"  Operation: DB: {row['DBName']}, Collection: {row['Collection']}, Type: {row['Operation Type']}\n")
                f.write(f"    Executions: {row['Executions']}\n")
                f.write(f"    Avg Duration: {row['Avg Duration(ms)']:.2f} ms\n")
                f.write(f"    Percentage of Total: {row['Percentage of Total (%)']:.2f}%\n")
                f.write("\n")
        else:
            f.write("  No operation type statistics available.\n\n")

        f.write("For complete details, refer to the generated Excel file.\n")

    print(f"Summary report generated at: '{summary_txt_path}'")

def main():
    # Enforce the command-line argument structure
    if len(sys.argv) != 3:
        print("Usage: python mongodb_log_analyzer.py <input_JSON_string> <output_excel_file>")
        print("Example: python mongodb_log_analyzer.py '{\"input\": \"path/to/your/input.csv\", \"startTimeStamp\": \"2023-01-01 00:00:00\", \"endTimeStamp\": \"2023-01-02 23:59:59\"}' output.xlsx")
        sys.exit(1)

    try:
        # Parse the JSON string from the first command-line argument
        inputs = json.loads(sys.argv[1])
        input_file_path = inputs['input']  # Required: path to the input log file
        output_file_path = sys.argv[2]      # Required: path to the output Excel file

        # Optional: startTimeStamp and endTimeStamp
        start_time_input = inputs.get('startTimeStamp', None)
        end_time_input = inputs.get('endTimeStamp', None)

    except json.JSONDecodeError:
        print("Error: Invalid JSON input for the first argument. Please provide a valid JSON string.")
        sys.exit(1)
    except KeyError:
        print("Error: Input JSON must contain an 'input' key specifying the CSV file path.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during argument parsing: {e}")
        sys.exit(1)

    # Validate input file existence
    if not os.path.isfile(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        sys.exit(1)

    # Convert timestamp strings to pandas datetime objects for filtering
    start_time = None
    end_time = None
    if start_time_input:
        try:
            start_time = pd.to_datetime(start_time_input)
        except ValueError:
            print(f"Warning: Could not parse start timestamp '{start_time_input}'. Time filtering will not apply start time.")
    if end_time_input:
        try:
            end_time = pd.to_datetime(end_time_input)
        except ValueError:
            print(f"Warning: Could not parse end timestamp '{end_time_input}'. Time filtering will not apply end time.")


    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
        total_lines_read = len(lines)
    except Exception as e:
        print(f"Error reading the input file '{input_file_path}': {e}")
        sys.exit(1)

    # Process all log lines and get aggregated DataFrames
    # Fixed: operation_stats_df and total_operations_processed_for_stats are now returned
    output_df, non_slow_query_df, error_df, query_stats_df, remote_ip_df, error_type_df, operation_stats_df, total_operations_processed_for_stats = \
        process_log_lines(lines, start_time, end_time)

    # Write results to Excel with various sheets
    # Fixed: operation_stats_df is now passed
    write_output_file(output_file_path, output_df, non_slow_query_df, error_df, query_stats_df, remote_ip_df, error_type_df, operation_stats_df)

    # Generate the summary report text file
    # Fixed: operation_stats_df is now passed, and total_operations_processed used for summary total
    generate_summary_txt(output_file_path, total_lines_read, total_operations_processed_for_stats,
                         output_df, query_stats_df, remote_ip_df, error_df, error_type_df, operation_stats_df)

if __name__ == "__main__":
    main()
