#!/usr/bin/env python3
import pandas as pd
import re
import json
from datetime import datetime
from collections import defaultdict
from statistics import mean

def normalize_query(query):
    """
    Normalizes a MySQL query by removing specific values and standardizing patterns.
    """
    # Remove specific values but keep the query structure
    normalized = re.sub(r"'[^']*'", "'?'", query)  # Replace string literals
    normalized = re.sub(r"\b\d+\b", "?", normalized)  # Replace numbers
    normalized = re.sub(r"IN \([^)]+\)", "IN (?)", normalized)  # Replace IN clause values
    normalized = re.sub(r"VALUES\s*\([^)]+\)", "VALUES (?)", normalized)  # Replace VALUES
    return normalized.strip()

def parse_time_str(time_str):
    """Parse MySQL slow query log timestamp format."""
    try:
        return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None

def process_mysql_log(file_path, start_time=None, end_time=None):
    """
    Process MySQL slow query log and extract key metrics.
    """
    # Initialize data structures
    queries = []
    current_query = {}
    query_stats = defaultdict(lambda: {
        "count": 0,
        "times": [],
        "rows_sent": [],
        "rows_examined": [],
        "lock_time": []
    })

    # Regex patterns
    time_pattern = r'# Time: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    user_pattern = r'# User@Host: ([^\[]+)\[([^\]]+)\]'
    query_stats_pattern = r'# Query_time: (\d+\.\d+)\s+Lock_time: (\d+\.\d+)\s+Rows_sent: (\d+)\s+Rows_examined: (\d+)'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # Match timestamp
        time_match = re.search(time_pattern, line)
        if time_match:
            if current_query:
                queries.append(current_query)
            current_query = {"timestamp": time_match.group(1)}
            continue

        # Match user info
        user_match = re.search(user_pattern, line)
        if user_match:
            current_query["user"] = user_match.group(1).strip()
            current_query["host"] = user_match.group(2).strip()
            continue

        # Match query stats
        stats_match = re.search(query_stats_pattern, line)
        if stats_match:
            current_query.update({
                "query_time": float(stats_match.group(1)),
                "lock_time": float(stats_match.group(2)),
                "rows_sent": int(stats_match.group(3)),
                "rows_examined": int(stats_match.group(4))
            })
            continue

        # If line starts with SET or SELECT or similar, it's the query
        if re.match(r'^(SELECT|UPDATE|DELETE|INSERT|SET|SHOW|CREATE|ALTER|DROP|CALL)', line):
            if "query" in current_query:
                current_query["query"] += " " + line
            else:
                current_query["query"] = line

    # Add the last query if exists
    if current_query:
        queries.append(current_query)

    # Process queries and create DataFrames
    processed_queries = []
    for q in queries:
        if "query" in q and "timestamp" in q:
            timestamp = parse_time_str(q["timestamp"])
            
            # Apply time filtering
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue

            normalized = normalize_query(q["query"])
            
            # Update query statistics
            query_stats[normalized]["count"] += 1
            query_stats[normalized]["times"].append(q.get("query_time", 0))
            query_stats[normalized]["rows_sent"].append(q.get("rows_sent", 0))
            query_stats[normalized]["rows_examined"].append(q.get("rows_examined", 0))
            query_stats[normalized]["lock_time"].append(q.get("lock_time", 0))

            processed_queries.append({
                "Timestamp": timestamp,
                "User": q.get("user", ""),
                "Host": q.get("host", ""),
                "Query Time (s)": q.get("query_time", 0),
                "Lock Time (s)": q.get("lock_time", 0),
                "Rows Sent": q.get("rows_sent", 0),
                "Rows Examined": q.get("rows_examined", 0),
                "Query": q["query"],
                "Normalized Query": normalized
            })

    # Create summary statistics
    summary_stats = []
    for query, stats in query_stats.items():
        summary_stats.append({
            "Normalized Query": query,
            "Executions": stats["count"],
            "Avg Query Time (s)": mean(stats["times"]),
            "Max Query Time (s)": max(stats["times"]),
            "Total Time (s)": sum(stats["times"]),
            "Avg Rows Sent": mean(stats["rows_sent"]),
            "Avg Rows Examined": mean(stats["rows_examined"]),
            "Rows Examined/Sent Ratio": mean(stats["rows_examined"]) / mean(stats["rows_sent"]) if mean(stats["rows_sent"]) > 0 else float('inf'),
            "Avg Lock Time (s)": mean(stats["lock_time"]),
            "Sample Query": next(q["Query"] for q in processed_queries if q["Normalized Query"] == query)
        })

    # Create DataFrames
    queries_df = pd.DataFrame(processed_queries)
    summary_df = pd.DataFrame(summary_stats)

    # Sort summary by total time descending
    summary_df = summary_df.sort_values("Total Time (s)", ascending=False)

    return queries_df, summary_df

def write_output_file(output_file_path, queries_df, summary_df):
    """Write the analysis results to an Excel file."""
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        # Write DataFrames to different sheets
        summary_df.to_excel(writer, sheet_name='Query Summary', index=False)
        queries_df.to_excel(writer, sheet_name='Detailed Queries', index=False)

        workbook = writer.book
        
        # Format the Query Summary sheet
        summary_sheet = writer.sheets['Query Summary']
        summary_sheet.autofilter(0, 0, len(summary_df), len(summary_df.columns) - 1)
        for idx, col in enumerate(summary_df.columns):
            max_len = max(
                summary_df[col].astype(str).str.len().max(),
                len(col)
            )
            summary_sheet.set_column(idx, idx, min(max_len + 2, 100))

        # Format the Detailed Queries sheet
        queries_sheet = writer.sheets['Detailed Queries']
        queries_sheet.autofilter(0, 0, len(queries_df), len(queries_df.columns) - 1)
        for idx, col in enumerate(queries_df.columns):
            max_len = max(
                queries_df[col].astype(str).str.len().max(),
                len(col)
            )
            queries_sheet.set_column(idx, idx, min(max_len + 2, 100))

def generate_summary_txt(output_excel_path, queries_df, summary_df):
    """Generate a text summary of the analysis."""
    summary_path = output_excel_path.replace('.xlsx', '_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MySQL Slow Query Log Analysis Summary\n")
        f.write("===================================\n\n")

        f.write(f"Total Queries Analyzed: {len(queries_df)}\n")
        f.write(f"Unique Query Patterns: {len(summary_df)}\n\n")

        f.write("Top 10 Most Time-Consuming Queries:\n")
        f.write("----------------------------------\n")
        top_queries = summary_df.head(10)
        for _, row in top_queries.iterrows():
            f.write(f"\nPattern: {row['Normalized Query'][:200]}...\n")
            f.write(f"Executions: {row['Executions']}\n")
            f.write(f"Average Time: {row['Avg Query Time (s)']:.3f}s\n")
            f.write(f"Total Time: {row['Total Time (s)']:.3f}s\n")
            f.write(f"Rows Examined/Sent Ratio: {row['Rows Examined/Sent Ratio']:.2f}\n")
            f.write("-" * 80 + "\n")

        f.write("\nOverall Statistics:\n")
        f.write("------------------\n")
        f.write(f"Total Query Time: {summary_df['Total Time (s)'].sum():.2f}s\n")
        f.write(f"Average Query Time: {summary_df['Avg Query Time (s)'].mean():.3f}s\n")
        f.write(f"Max Query Time: {summary_df['Max Query Time (s)'].max():.3f}s\n")

def main(inputs):
    """Main function to process MySQL slow query log."""
    log_file = inputs.get('log_file')
    start_time = inputs.get('start_datetime')
    end_time = inputs.get('end_datetime')

    # Process the log file
    queries_df, summary_df = process_mysql_log(log_file, start_time, end_time)
    
    # Generate Excel report
    write_output_file(log_file + '.xlsx', queries_df, summary_df)
    
    # Generate text summary
    generate_summary_txt(log_file + '.xlsx', queries_df, summary_df)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mysql_log_parser.py <log_file>")
        sys.exit(1)
    
    main({"log_file": sys.argv[1]})
