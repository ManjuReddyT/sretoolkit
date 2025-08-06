import streamlit as st
import pandas as pd
import re
import os
from io import StringIO

def parse_strace_log(log_content):
    """
    Parses an strace log file and extracts syscall information.
    """
    # Regex to capture syscall, arguments, return value, and time
    # This regex is a starting point and may need to be refined
    line_regex = re.compile(r'^(?P<syscall>\w+)\((?P<args>.*?)\)\s+=\s+(?P<retval>-?\d+|0x[a-fA-F0-9]+)\s*(?P<error>\w+)?\s*.*?\s+<(?P<time>\d+\.\d+)>$')

    parsed_data = []
    for line in log_content.splitlines():
        match = line_regex.match(line)
        if match:
            data = match.groupdict()
            parsed_data.append({
                'syscall': data['syscall'],
                'args': data['args'],
                'retval': data['retval'],
                'error': data['error'] if data['error'] else '',
                'time': float(data['time'])
            })
    return pd.DataFrame(parsed_data)

def analyze_syscall_frequency(df):
    """
    Analyzes the frequency and total time of each syscall.
    """
    if df.empty:
        return pd.DataFrame()

    frequency = df.groupby('syscall').agg(
        count=('syscall', 'size'),
        total_time=('time', 'sum')
    ).sort_values(by='count', ascending=False)
    return frequency

def analyze_errors(df):
    """
    Filters for syscalls that returned an error.
    """
    if df.empty:
        return pd.DataFrame()

    return df[df['error'] != '']

def detect_language(df):
    """
    Detects the programming language based on syscall patterns.
    """
    if df.empty:
        return "Unknown", "No specific language patterns detected."

    syscalls = set(df['syscall'])
    file_accesses = df[df['syscall'].isin(['open', 'openat'])]['args'].str.cat(sep='\n')

    lang_patterns = {
        "Java": {
            "files": [r'\.jar', r'\.class', 'libjvm.so'],
            "syscalls": ['mmap', 'munmap', 'futex']
        },
        "Python": {
            "files": [r'\.py', r'\.pyc', r'\.so'],
            "syscalls": ['openat', 'lseek']
        },
        "Node.js": {
            "files": ['node_modules', 'libnode.so', 'libv8.so'],
            "syscalls": ['epoll_wait', 'read', 'write']
        }
    }

    detected_langs = []
    for lang, patterns in lang_patterns.items():
        file_match = any(re.search(p, file_accesses) for p in patterns['files'])
        syscall_match = all(s in syscalls for s in patterns['syscalls'])
        if file_match and syscall_match:
            detected_langs.append(lang)

    if detected_langs:
        return detected_langs[0], f"Detected patterns consistent with {detected_langs[0]}."
    else:
        return "Unknown", "Could not confidently determine the language."

def analyze_io(df):
    """
    Analyzes I/O related syscalls.
    """
    if df.empty:
        return pd.DataFrame()

    io_syscalls = ['read', 'write', 'open', 'openat', 'close', 'lseek', 'pread64', 'pwrite64', 'readv', 'writev', 'sendfile', 'stat', 'fstat', 'lstat', 'poll', 'epoll_wait', 'select']
    io_df = df[df['syscall'].isin(io_syscalls)]

    if io_df.empty:
        return pd.DataFrame()

    return io_df.groupby('syscall').agg(
        count=('syscall', 'size'),
        total_time=('time', 'sum')
    ).sort_values(by='count', ascending=False)

def analyze_process_management(df):
    """
    Analyzes process management related syscalls.
    """
    if df.empty:
        return pd.DataFrame()

    proc_syscalls = ['fork', 'vfork', 'clone', 'execve', 'exit', 'exit_group', 'wait4', 'waitpid', 'kill', 'tkill', 'futex']
    proc_df = df[df['syscall'].isin(proc_syscalls)]

    if proc_df.empty:
        return pd.DataFrame()

    return proc_df.groupby('syscall').agg(
        count=('syscall', 'size'),
        total_time=('time', 'sum')
    ).sort_values(by='count', ascending=False)


def main(inputs):
    """
    Main function for the strace analyzer.
    """
    log_file = inputs.get("log_file")

    if log_file:
        try:
            # To handle multiple file uploads
            if isinstance(log_file, list):
                log_content = ""
                for uploaded_file in log_file:
                    log_content += uploaded_file.read().decode('utf-8')
            else:
                log_content = log_file.read().decode('utf-8')

            df = parse_strace_log(log_content)

            if df.empty:
                st.warning("No valid strace lines found in the log file. Please check the file format.")
                return

            st.header("Strace Analysis")

            # Allow downloading the parsed data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Parsed Data as CSV",
                csv,
                "strace_parsed.csv",
                "text/csv",
                key='download-csv'
            )

            # Display a preview of the parsed data
            with st.expander("Parsed Data Preview"):
                st.dataframe(df.head())

            # Syscall Frequency and Duration
            st.subheader("Syscall Frequency and Duration")
            freq_df = analyze_syscall_frequency(df)
            st.dataframe(freq_df)

            # Error Analysis
            st.subheader("Error Analysis")
            error_df = analyze_errors(df)
            if not error_df.empty:
                st.dataframe(error_df)
            else:
                st.success("No errors found in the syscalls.")

            # Language Specific Analysis
            st.subheader("Language Specific Analysis")
            lang, reason = detect_language(df)
            st.info(f"Detected Language: **{lang}**")
            st.write(reason)

            # I/O Analysis
            st.subheader("I/O Analysis")
            io_df = analyze_io(df)
            if not io_df.empty:
                st.dataframe(io_df)
            else:
                st.info("No I/O related syscalls found.")

            # Process Management Analysis
            st.subheader("Process Management Analysis")
            proc_df = analyze_process_management(df)
            if not proc_df.empty:
                st.dataframe(proc_df)
            else:
                st.info("No process management related syscalls found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an strace log file to begin analysis.")

if __name__ == '__main__':
    # This part is for local testing and will not be executed by the Streamlit app
    # You can create a dummy inputs dictionary to test the script
    # For example:
    # from io import BytesIO
    # sample_log = b'openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3 <0.000008>\n'
    # inputs = {"log_file": BytesIO(sample_log)}
    # main(inputs)
    pass
