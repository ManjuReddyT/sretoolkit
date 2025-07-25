import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
from datetime import datetime
import io
import zipfile
import tarfile
import gzip
import os
import tempfile
import json
import requests  # Ensure requests is imported for AI analysis


# AI Analysis Function (reused from other scripts for consistency)
def run_ai_analysis(log_snippet, ollama_api_url, ollama_model_name, prompt_template_config):
    st.subheader("🤖 AI Analysis")
    with st.spinner("Running AI analysis..."):
        try:
            prompt = build_ai_prompt(prompt_template_config, log_snippet)

            headers = {'Content-Type': 'application/json'}
            data = {
                "model": ollama_model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": 2048}
            }

            response = requests.post(ollama_api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            ai_response_placeholder = st.empty()
            full_ai_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    try:
                        json_data = json.loads(chunk.decode('utf-8'))
                        full_ai_response += json_data.get("response", "")
                        ai_response_placeholder.markdown(full_ai_response)
                    except json.JSONDecodeError:
                        pass

            st.success("AI analysis complete!")
            return full_ai_response
        except requests.exceptions.ConnectionError:
            st.error("AI analysis failed: Could not connect to Ollama API. Ensure Ollama is running and accessible.")
            st.info(f"Attempted to connect to: `{ollama_api_url}` with model: `{ollama_model_name}`")
        except requests.exceptions.RequestException as e:
            st.error(f"AI analysis failed due to a request error: {e}")
            st.info(f"Check Ollama server logs for details. Status Code: {e.response.status_code if e.response else 'N/A'}")
        except Exception as e:
            st.error(f"An unexpected error occurred during AI analysis: {e}")
    return "AI analysis could not be completed."


# Helper to build AI prompt (replicated for consistency)
def build_ai_prompt(prompt_template, content):
    system_role = prompt_template.get("system_role", "You are an expert system performance analyst.")
    task_desc = prompt_template.get("task_description", "Please analyze this data summary:")
    analysis_points = prompt_template.get("analysis_points", [
        'Identify key insights.',
        'Suggest areas for optimization.'
    ])
    output_format = prompt_template.get("output_format", "Provide your analysis in clear, organized paragraphs.")
    
    if not isinstance(analysis_points, list):
        analysis_points = ["Provide key insights and recommendations.", "Suggest areas for optimization based on thread states and lock contention."]

    analysis_points_str = '\n'.join(f'- {point}' for point in analysis_points)

    return (
        f"{system_role}\n\n"
        f"{task_desc}\n\n"
        "Here is the relevant data summary:\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Please analyze this data focusing on:\n"
        f"{analysis_points_str}\n\n"
        f"{output_format}"
    )


class ThreadDumpAnalyzer:
    def __init__(self):
        self.threads = []
        self.deadlocks = []
        self.summary = {}
        self.file_info = {}

    def parse_multiple_files(self, files_content):
        """Parse multiple thread dump files"""
        all_threads = []
        all_deadlocks = []
        file_summaries = {}

        for filename, content in files_content.items():
            # Parse individual file
            temp_analyzer = ThreadDumpAnalyzer()
            temp_analyzer.parse_thread_dump(content)

            # Store file-specific data
            file_summaries[filename] = {
                'threads': len(temp_analyzer.threads),
                'deadlocks': len(temp_analyzer.deadlocks),
                'states': Counter(t['state'] for t in temp_analyzer.threads),
                'daemon_count': sum(1 for t in temp_analyzer.threads if t['daemon'])
            }

            # Add file info to each thread
            for thread in temp_analyzer.threads:
                thread['source_file'] = filename

            all_threads.extend(temp_analyzer.threads)
            all_deadlocks.extend(temp_analyzer.deadlocks)

        self.threads = all_threads
        self.deadlocks = all_deadlocks
        self.file_info = file_summaries
        self._generate_summary()

    def parse_thread_dump(self, content):
        """Parse thread dump content and extract thread information"""
        self.threads = []
        self.deadlocks = []

        # Split by thread entries (lines starting with quotes)
        # Handles cases where a thread block might not start with a quote but previous one ended cleanly
        thread_blocks = re.split(r'\n(?="[^"]+")', content)  # More robust split

        for block in thread_blocks:
            if not block.strip():
                continue

            thread_info = self._parse_thread_block(block)
            if thread_info:
                self.threads.append(thread_info)

        # Detect deadlocks
        self._detect_deadlocks()

        # Generate summary
        self._generate_summary()

    def _parse_thread_block(self, block):
        """Parse individual thread block"""
        lines = block.strip().split('\n')
        if not lines:
            return None

        thread_info = {
            'name': 'Unknown',
            'id': 'Unknown',
            'state': 'Unknown',
            'priority': 'Unknown',
            'daemon': False,
            'stack_trace': [],
            'locks_held': [],
            'locks_waiting': [],
            'raw_block': block,
            'source_file': 'single_file'  # Default for single file analysis
        }

        # Parse thread header (first line)
        header = lines[0]

        # Extract thread name
        name_match = re.search(r'"([^"]+)"', header)
        if name_match:
            thread_info['name'] = name_match.group(1)

        # Extract thread ID (tid)
        id_match = re.search(r'tid=0x([0-9a-fA-F]+)', header)
        if id_match:
            thread_info['id'] = id_match.group(1)
        else:  # Fallback to #ID
            id_match = re.search(r'#(\d+)', header)
            if id_match:
                thread_info['id'] = id_match.group(1)

        # Extract priority
        prio_match = re.search(r'prio=(\d+)', header)
        if prio_match:
            thread_info['priority'] = prio_match.group(1)

        # Check if daemon
        if 'daemon' in header.lower():
            thread_info['daemon'] = True

        # Parse thread state and stack trace
        in_stack = False
        for line in lines[1:]:  # Start from the second line
            line = line.strip()

            if 'java.lang.Thread.State:' in line:
                state_match = re.search(r'java\.lang\.Thread\.State:\s*(\w+)', line)
                if state_match:
                    thread_info['state'] = state_match.group(1)
                in_stack = True  # Stack trace usually follows the state line
                continue  # Process next line for stack

            if line.startswith('at '):
                thread_info['stack_trace'].append(line)
                in_stack = True
            elif line.startswith('- locked <'):  # Example: - locked <0x000000076ab62218> (a java.lang.Object)
                lock_match = re.search(r'- locked <([^>]+)>', line)
                if lock_match:
                    thread_info['locks_held'].append(lock_match.group(1))
            elif line.startswith('- waiting on <'):  # Example: - waiting on <0x000000076ab62208> (a java.lang.Object)
                lock_match = re.search(r'- waiting on <([^>]+)>', line)
                if lock_match:
                    thread_info['locks_waiting'].append(lock_match.group(1))
            elif line.startswith('- parking to wait for'):  # Specific for WAITING/TIMED_WAITING
                lock_match = re.search(r'parking to wait for\s*<([^>]+)>', line)
                if lock_match:
                    thread_info['locks_waiting'].append(lock_match.group(1))

        return thread_info

    def _detect_deadlocks(self):
        """Detect potential deadlocks based on circular wait conditions."""
        self.deadlocks = []
        lock_owners = {}  # {lock_id: thread_name}
        thread_waiting_for = {}  # {thread_name: lock_id_it_waits_for}

        for thread in self.threads:
            # A thread can hold multiple locks, but for simple detection, we consider its primary held locks
            for lock_held in thread['locks_held']:
                lock_owners[lock_held] = thread['name']

            # A thread can wait for multiple locks, but we simplify to one for detection
            if thread['locks_waiting']:
                thread_waiting_for[thread['name']] = thread['locks_waiting'][0]  # Assume waiting for the first lock

        # Check for cycles
        for thread_name, waiting_lock_id in thread_waiting_for.items():
            if waiting_lock_id in lock_owners:
                owner_thread_name = lock_owners[waiting_lock_id]

                # Check if the owner thread is also waiting for a lock held by the initial thread
                if owner_thread_name in thread_waiting_for:
                    owner_waiting_lock_id = thread_waiting_for[owner_thread_name]

                    if owner_waiting_lock_id in lock_owners and lock_owners[owner_waiting_lock_id] == thread_name:
                        # Deadlock detected!
                        deadlock_found = {
                            'thread1': thread_name,
                            'thread1_waiting_for_lock': waiting_lock_id,
                            'thread2': owner_thread_name,
                            'thread2_waiting_for_lock': owner_waiting_lock_id
                        }
                        # Add only if not already detected (due to symmetric nature)
                        if deadlock_found not in self.deadlocks and {
                            'thread1': owner_thread_name,
                            'thread1_waiting_for_lock': owner_waiting_lock_id,
                            'thread2': thread_name,
                            'thread2_waiting_for_lock': waiting_lock_id
                        } not in self.deadlocks:
                            self.deadlocks.append(deadlock_found)

    def _generate_summary(self):
        """Generate summary statistics"""
        total_threads = len(self.threads)
        state_counts = Counter(thread['state'] for thread in self.threads)
        daemon_count = sum(1 for thread in self.threads if thread['daemon'])

        # Find threads with most locks held
        lock_holders_counts = [(thread['name'], len(thread['locks_held']))
                               for thread in self.threads if thread['locks_held']]
        lock_holders_counts.sort(key=lambda x: x[1], reverse=True)

        # Find threads with most locks waiting
        lock_waiters_counts = [(thread['name'], len(thread['locks_waiting']))
                               for thread in self.threads if thread['locks_waiting']]
        lock_waiters_counts.sort(key=lambda x: x[1], reverse=True)

        # Find most common stack traces (using the top frame)
        stack_patterns = Counter()
        for thread in self.threads:
            if thread['stack_trace']:
                # Use top method as pattern
                top_method = thread['stack_trace'][0]
                # Extract class.method name
                method_match = re.search(r'at ([^(]+)', top_method)
                if method_match:
                    method_name = method_match.group(1).strip()
                    stack_patterns[method_name] += 1

        # Thread name patterns (e.g., "http-nio", "pool-")
        name_patterns = Counter()
        for thread in self.threads:
            name = thread['name']
            if '-' in name:
                pattern = name.split('-')[0].strip()
                name_patterns[pattern] += 1
            elif 'pool-' in name:  # Catch generic pools
                pattern = name.split('pool-')[0] + 'pool' if name.startswith('pool-') else 'pool'
                name_patterns[pattern] += 1
            else:
                name_patterns[name.split(' ')[0].strip()] += 1  # Take first word for general threads

        self.summary = {
            'total_threads': total_threads,
            'state_distribution': dict(state_counts),
            'daemon_threads': daemon_count,
            'deadlocks_found': len(self.deadlocks),
            'deadlock_details': self.deadlocks,  # Include details of detected deadlocks
            'top_lock_holders': lock_holders_counts[:10],
            'top_lock_waiters': lock_waiters_counts[:10],
            'common_stack_patterns': stack_patterns.most_common(10),
            'thread_name_patterns': name_patterns.most_common(10)
        }
        
        # Generate a textual summary for AI analysis
        summary_text_parts = [
            f"**Thread Dump Analysis Summary**",
            f"Total threads: {self.summary['total_threads']}",
            f"Daemon threads: {self.summary['daemon_threads']}",
            f"Deadlocks detected: {self.summary['deadlocks_found']}"
        ]
        
        if self.summary['state_distribution']:
            summary_text_parts.append("\n**Thread State Distribution:**")
            for state, count in self.summary['state_distribution'].items():
                summary_text_parts.append(f"- {state}: {count} threads")
        
        if self.summary['top_lock_holders']:
            summary_text_parts.append("\n**Top Lock Holders:**")
            for name, count in self.summary['top_lock_holders']:
                summary_text_parts.append(f"- Thread '{name}' holds {count} locks.")

        if self.summary['top_lock_waiters']:
            summary_text_parts.append("\n**Top Lock Waiters:**")
            for name, count in self.summary['top_lock_waiters']:
                summary_text_parts.append(f"- Thread '{name}' is waiting for {count} locks.")

        if self.summary['common_stack_patterns']:
            summary_text_parts.append("\n**Common Stack Patterns (Top 10 Methods):**")
            for method, count in self.summary['common_stack_patterns']:
                summary_text_parts.append(f"- `{method}`: {count} occurrences.")

        if self.summary['thread_name_patterns']:
            summary_text_parts.append("\n**Common Thread Name Patterns:**")
            for pattern, count in self.summary['thread_name_patterns']:
                summary_text_parts.append(f"- `{pattern}`: {count} threads.")

        if self.summary['deadlock_details']:
            summary_text_parts.append("\n**Deadlock Details:**")
            for dl in self.summary['deadlock_details']:
                summary_text_parts.append(
                    f"- Thread '{dl['thread1']}' waits for lock `{dl['thread1_waiting_for_lock']}` held by '{dl['thread2']}' "
                    f"which in turn waits for lock `{dl['thread2_waiting_for_lock']}` held by '{dl['thread1']}'."
                )
        
        self.summary['Summary_Text'] = "\n".join(summary_text_parts)


def extract_files_from_archive(uploaded_file):
    """Extract files from ZIP, TAR, or GZ archives. Returns a dictionary: {filename: content_string}"""
    files_content = {}

    try:
        file_bytes = uploaded_file.read()
        filename_lower = uploaded_file.name.lower()

        if filename_lower.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Skip directories and non-text-like files
                    if file_info.is_dir():
                        continue
                    # Check for common text/log extensions or no extension for raw dumps
                    if not any(file_info.filename.lower().endswith(ext) for ext in
                               ('.txt', '.log', '.dump', '.out')) and '.' in file_info.filename:
                        continue  # Skip binary or other file types

                    try:
                        content = zip_ref.read(file_info.filename).decode('utf-8', errors='ignore')
                        files_content[file_info.filename] = content
                    except Exception as e:
                        st.warning(f"Could not read {file_info.filename} from ZIP: {e}")

        elif filename_lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
            mode = 'r:*'  # Automatically determine compression
            with tarfile.open(fileobj=io.BytesIO(file_bytes), mode=mode) as tar_ref:
                for member in tar_ref.getmembers():
                    if member.isfile():
                        if not any(member.name.lower().endswith(ext) for ext in
                                   ('.txt', '.log', '.dump', '.out')) and '.' in member.name:
                            continue  # Skip binary or other file types

                        file_obj = tar_ref.extractfile(member)
                        if file_obj:
                            try:
                                content = file_obj.read().decode('utf-8', errors='ignore')
                                files_content[member.name] = content
                            except Exception as e:
                                st.warning(f"Could not read {member.name} from TAR: {e}")
                            finally:
                                file_obj.close()

        elif filename_lower.endswith('.gz'):
            try:
                content = gzip.decompress(file_bytes).decode('utf-8', errors='ignore')
                files_content[uploaded_file.name.replace('.gz', '')] = content
            except Exception as e:
                st.warning(f"Could not decompress {uploaded_file.name}: {e}")
        elif filename_lower.endswith(('.txt', '.log', '.dump', '.out')):
            # Regular text file
            content = file_bytes.decode('utf-8', errors='ignore')
            files_content[uploaded_file.name] = content
        else:
            st.warning(
                f"Unsupported file type: {uploaded_file.name}. Only common text/log, ZIP, TAR, GZ archives are supported.")

    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    return files_content


def create_advanced_charts(analyzer):
    """Create comprehensive charts for thread dump analysis"""

    # 1. Thread State Distribution with Multiple Files (if applicable)
    if len(analyzer.file_info) > 1:
        st.subheader("📊 Thread States by File")

        file_state_data = []
        for filename, info in analyzer.file_info.items():
            for state, count in info['states'].items():
                file_state_data.append({
                    'File': filename,
                    'State': state,
                    'Count': count
                })

        if file_state_data:
            df_states = pd.DataFrame(file_state_data)
            fig = px.bar(df_states, x='File', y='Count', color='State',
                         title="Thread States Distribution Across Files",
                         barmode='stack', text='Count')
            fig.update_xaxes(tickangle=45, title_text="")  # Rotate labels for better readability
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No thread state data available for file comparison.")

    # 2. Overall Thread State Pie Chart
    st.subheader("🥧 Overall Thread State Distribution")
    if analyzer.summary.get('state_distribution'):
        df_overall_states = pd.DataFrame([
            {'State': state, 'Count': count}
            for state, count in analyzer.summary['state_distribution'].items()
        ])
        fig = px.pie(
            df_overall_states,
            values='Count',
            names='State',
            title="Overall Thread States Distribution",
            hole=0.3  # Donut chart
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No overall thread state data to display.")

    # 3. Thread Name Patterns
    st.subheader("🏷️ Thread Pool/Name Patterns")
    if analyzer.summary.get('thread_name_patterns'):
        patterns_df = pd.DataFrame(
            analyzer.summary['thread_name_patterns'],
            columns=['Pattern', 'Count']
        )
        fig = px.bar(patterns_df, x='Count', y='Pattern', orientation='h',
                     title="Most Common Thread Name Patterns (Top 10)",
                     text='Count')
        fig.update_layout(yaxis={'autorange': 'reversed'})  # Highest count at top
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common thread name patterns found.")

    # 4. Lock Analysis (Scatter plot for held vs. waiting)
    st.subheader("🔒 Lock Contention Analysis")
    lock_data = []
    for thread in analyzer.threads:
        lock_data.append({
            'Thread': thread['name'],
            'Locks Held': len(thread['locks_held']),
            'Locks Waiting': len(thread['locks_waiting']),
            'State': thread['state'],
            'Source File': thread['source_file']
        })

    if lock_data:
        lock_df = pd.DataFrame(lock_data)

        # Filter for threads involved in any locking to make plot relevant
        lock_df_filtered = lock_df[(lock_df['Locks Held'] > 0) | (lock_df['Locks Waiting'] > 0)]

        if not lock_df_filtered.empty:
            fig = px.scatter(lock_df_filtered, x='Locks Held', y='Locks Waiting',
                             color='State', hover_data=['Thread', 'Source File', 'Locks Held', 'Locks Waiting'],
                             title="Thread Lock Contention: Held vs. Waiting",
                             labels={'Locks Held': 'Number of Locks Held',
                                     'Locks Waiting': 'Number of Locks Waiting For'},
                             size=[(c + 1) * 5 for c in
                                   lock_df_filtered['Locks Held'] + lock_df_filtered['Locks Waiting']]
                             # Size by total locks
                             )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threads found holding or waiting for locks.")
    else:
        st.info("No lock data available for analysis.")

    # 5. Thread Priority Distribution
    st.subheader("⚡ Thread Priority Distribution")
    priority_counts = Counter(str(thread['priority']) for thread in analyzer.threads if thread['priority'] != 'Unknown')
    if priority_counts:
        priority_df = pd.DataFrame([
            {'Priority': prio, 'Count': count}
            for prio, count in priority_counts.items()
        ])
        fig = px.bar(priority_df, x='Priority', y='Count',
                     title="Thread Priority Distribution",
                     labels={'Priority': 'Thread Priority', 'Count': 'Number of Threads'},
                     text='Count')
        fig.update_xaxes(type='category')  # Treat priorities as categories
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No thread priority data available.")

    # 6. Stack Trace Patterns (Top methods)
    st.subheader("📚 Common Stack Trace Patterns")
    if analyzer.summary.get('common_stack_patterns'):
        stack_df = pd.DataFrame(
            analyzer.summary['common_stack_patterns'][:15],  # Top 15
            columns=['Method', 'Count']
        )
        fig = px.bar(stack_df, x='Count', y='Method', orientation='h',
                     title="Most Common Methods in Stack Traces (Top 15)",
                     text='Count')
        fig.update_layout(yaxis={'autorange': 'reversed'})  # Highest count at top
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common stack trace patterns found.")

    # 7. File Comparison (if multiple files)
    if len(analyzer.file_info) > 1:
        st.subheader("📁 File Comparison: Key Metrics")

        comparison_data = []
        for filename, info in analyzer.file_info.items():
            comparison_data.append({
                'File': filename,
                'Total Threads': info['threads'],
                'Deadlocks': info['deadlocks'],
                'Blocked Threads': info['states'].get('BLOCKED', 0),
                'Waiting Threads': info['states'].get('WAITING', 0),
                'Timed Waiting Threads': info['states'].get('TIMED_WAITING', 0),
                'Runnable Threads': info['states'].get('RUNNABLE', 0),
                'Daemon Threads': info['daemon_count']
            })

        comp_df = pd.DataFrame(comparison_data)

        # Multi-metric comparison with tabs for better organization
        st.markdown("Select a metric to compare across files:")
        metric_options = ['Total Threads', 'Deadlocks', 'Blocked Threads', 'Waiting Threads', 'Timed Waiting Threads',
                          'Runnable Threads', 'Daemon Threads']
        selected_metric = st.selectbox("Choose Metric", metric_options)

        fig_comp = px.bar(comp_df, x='File', y=selected_metric,
                          title=f'{selected_metric} Across Files',
                          text=selected_metric)
        fig_comp.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.subheader("Detailed File Comparison Table")
        st.dataframe(comp_df.set_index('File'))  # Display as a table too
    else:
        st.info("Upload multiple thread dump files or an archive containing them to see file comparison charts.")


def save_results_to_excel(analyzer, output_file_path, aianalysis_enabled, ai_analysis_result):
    """
    Saves the detailed thread data and summary to an Excel file.
    Includes AI analysis if enabled and available.
    """
    try:
        with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
            # Sheet 1: All Threads Detail
            threads_df = pd.DataFrame(analyzer.threads)
            # Exclude raw_block for Excel clarity if it's very large
            if 'raw_block' in threads_df.columns:
                threads_df = threads_df.drop(columns=['raw_block'])
            threads_df.to_excel(writer, sheet_name='All Threads Detail', index=False)

            # Sheet 2: Summary
            summary_data = {
                'Metric': [],
                'Value': []
            }
            summary_data['Metric'].append('Total Threads')
            summary_data['Value'].append(analyzer.summary.get('total_threads', 0))
            summary_data['Metric'].append('Daemon Threads')
            summary_data['Value'].append(analyzer.summary.get('daemon_threads', 0))
            summary_data['Metric'].append('Deadlocks Found')
            summary_data['Value'].append(analyzer.summary.get('deadlocks_found', 0))

            for state, count in analyzer.summary.get('state_distribution', {}).items():
                summary_data['Metric'].append(f'Threads in State: {state}')
                summary_data['Value'].append(count)

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 3: Deadlock Details
            if analyzer.deadlocks:
                deadlocks_df = pd.DataFrame(analyzer.deadlocks)
                deadlocks_df.to_excel(writer, sheet_name='Deadlock Details', index=False)
            else:
                empty_df = pd.DataFrame([["No deadlocks detected"]], columns=["Info"])
                empty_df.to_excel(writer, sheet_name='Deadlock Details', index=False)

            # Sheet 4: Top Lock Holders
            if analyzer.summary.get('top_lock_holders'):
                lock_holders_df = pd.DataFrame(analyzer.summary['top_lock_holders'],
                                               columns=['Thread Name', 'Locks Held Count'])
                lock_holders_df.to_excel(writer, sheet_name='Top Lock Holders', index=False)

            # Sheet 5: Top Lock Waiters
            if analyzer.summary.get('top_lock_waiters'):
                lock_waiters_df = pd.DataFrame(analyzer.summary['top_lock_waiters'],
                                               columns=['Thread Name', 'Locks Waiting Count'])
                lock_waiters_df.to_excel(writer, sheet_name='Top Lock Waiters', index=False)

            # Sheet 6: Common Stack Patterns
            if analyzer.summary.get('common_stack_patterns'):
                stack_patterns_df = pd.DataFrame(analyzer.summary['common_stack_patterns'],
                                                 columns=['Method Pattern', 'Count'])
                stack_patterns_df.to_excel(writer, sheet_name='Common Stack Patterns', index=False)

            # Sheet 7: Thread Name Patterns
            if analyzer.summary.get('thread_name_patterns'):
                name_patterns_df = pd.DataFrame(analyzer.summary['thread_name_patterns'],
                                                columns=['Name Pattern', 'Count'])
                name_patterns_df.to_excel(writer, sheet_name='Thread Name Patterns', index=False)

            # New Sheet: AI Analysis (only if enabled and result is available)
            if aianalysis_enabled and ai_analysis_result:
                ai_df = pd.DataFrame({"AI Analysis Report": [ai_analysis_result]})
                ai_df.to_excel(writer, sheet_name='AI Analysis', index=False)

            st.success(f"Analysis results saved to Excel: {output_file_path}")
            return True
    except Exception as e:
        st.error(f"Error saving results to Excel: {e}")
        return False


def main():
    st.subheader("🔍 Advanced Thread Dump Analyzer")
    st.markdown("Upload thread dump files or paste raw content to analyze thread states, identify deadlocks, and detect performance bottlenecks.")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ThreadDumpAnalyzer()
    
    # Initialize session state for raw input and processed status
    if 'raw_thread_dump_content' not in st.session_state:
        st.session_state.raw_thread_dump_content = ""
    if 'analysis_performed' not in st.session_state:
        st.session_state.analysis_performed = False
    if 'ai_analysis_result' not in st.session_state:
        st.session_state.ai_analysis_result = None # Initialize AI analysis result storage

    # AI Config Loading (from session state, provided by runpage.py)
    ai_config = st.session_state.get("config", {})
    ollama_api_url = ai_config.get("ollama_api_url", "http://localhost:11434/api/generate")
    ollama_model_name = ai_config.get("ollama_model_name", "deepseek-r1:8b")
    aianalysis_enabled = ai_config.get("aianalysis", False)

    # Define a robust default prompt template config for thread dump analysis
    default_prompt_template_config = {
        "system_role": "You are an expert in Java application performance, JVM internals, and troubleshooting, specializing in analyzing thread dump output. Your task is to analyze the provided thread dump summary and identify potential issues, bottlenecks, and suggest targeted solutions.",
        "task_description": "Analyze this thread dump summary. Identify common thread states, lock contention, potential deadlocks, and frequent code paths. Explain what these patterns might indicate about the application's health and performance.",
        "analysis_points": [
            "Summarize the overall health of the application based on thread states (e.g., high BLOCKED/WAITING threads vs. high RUNNABLE).",
            "Identify and elaborate on any detected deadlocks, including the threads and locks involved.",
            "Discuss potential causes for high numbers of threads in BLOCKED, WAITING, or TIMED_WAITING states, and suggest areas for investigation (e.g., database calls, external service calls, I/O operations, synchronized blocks).",
            "Highlight the most common thread pool patterns and their significance.",
            "Analyze lock contention by identifying threads holding many locks or waiting for heavily contended locks. Suggest strategies to reduce contention.",
            "Point out critical or problematic methods appearing frequently in stack traces, especially those in RUNNABLE threads that might indicate CPU-bound tasks or long-running operations.",
            "Propose actionable recommendations to resolve identified issues and optimize application performance. Consider code-level changes, configuration tuning, or infrastructure adjustments."
        ],
        "output_format": "Provide a comprehensive, structured analysis in markdown format. Start with an **Executive Summary** highlighting the most critical findings. Follow with detailed findings for each category (e.g., Thread States, Deadlocks, Lock Contention, Common Code Paths). Conclude with clear, prioritized, and actionable **Recommendations** for performance improvement."
    }
    prompt_template_config = ai_config.get("ai_prompt", default_prompt_template_config)

    # Input Section: Upload or Paste
    tab1, tab2 = st.tabs(["📂 Upload Thread Dumps", "📋 Paste Raw Thread Dump"])

    uploaded_files = None
    raw_text_input = ""
    analysis_trigger = False # Flag to trigger analysis

    with tab1:
        uploaded_files = st.file_uploader(
            "Choose thread dump file(s) or archive(s)",
            type=['txt', 'log', 'dump', 'out', 'zip', 'tar', 'gz', 'tgz'],
            accept_multiple_files=True,
            key="file_uploader",
            help="Upload one or more thread dump files or a single archive containing multiple dumps (ZIP, TAR, GZ).",
            on_change=lambda: [setattr(st.session_state, 'analysis_performed', False), setattr(st.session_state, 'ai_analysis_result', None)] # Reset on new upload
        )
        if uploaded_files:
            if st.button("Analyze Uploaded Dumps", key='analyze_uploaded'):
                analysis_trigger = True
                st.session_state.analysis_performed = False # Ensure re-analysis on button click
                st.session_state.ai_analysis_result = None # Clear previous AI result
                st.session_state.raw_thread_dump_content = "" # Clear raw input if files are uploaded

    with tab2:
        st.session_state.raw_thread_dump_content = st.text_area(
            "Paste your raw thread dump content here:",
            value=st.session_state.raw_thread_dump_content,
            height=300,
            key="raw_text_area",
            on_change=lambda: [setattr(st.session_state, 'analysis_performed', False), setattr(st.session_state, 'ai_analysis_result', None)] # Reset on paste change
        )
        col_sample_button, col_analyze_raw = st.columns([1, 2])
        with col_sample_button:
            if st.button("Load Sample Data", key='load_sample_data'):
                sample_data = '''
"main" #1 prio=5 os_prio=0 tid=0x00007f8a2800b800 nid=0x1234 waiting on condition [0x00007f8a37ffe000]
   java.lang.Thread.State: WAITING (parking)
    at sun.misc.Unsafe.park(Native Method)
    at java.util.concurrent.locks.LockSupport.park(LockSupport.java:175)
    at java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.await(AbstractQueuedSynchronizer.java:2039)
    - parking to wait for <0x000000076ab62208> (a java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject)

"http-nio-8080-exec-1" #10 prio=5 os_prio=0 tid=0x00007f8a28001000 nid=0x1235 waiting for monitor entry [0x00007f8a37ffd000]
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.DeadlockExample.method1(DeadlockExample.java:25)
    - waiting to lock <0x000000076ab62208> (a java.lang.Object)
    - locked <0x000000076ab62218> (a java.lang.Object)

"http-nio-8080-exec-2" #11 prio=5 os_prio=0 tid=0x00007f8a28002000 nid=0x1236 waiting for monitor entry [0x00007f8a37ffc000]
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.DeadlockExample.method2(DeadlockExample.java:35)
    - waiting to lock <0x000000076ab62218> (a java.lang.Object)
    - locked <0x000000076ab62208> (a java.lang.Object)

"pool-1-thread-1" #12 prio=5 os_prio=0 tid=0x00007f8a28003000 nid=0x1237 runnable [0x00007f8a37ffb000]
   java.lang.Thread.State: RUNNABLE
    at java.io.FileInputStream.readBytes(Native Method)
    at java.io.FileInputStream.read(FileInputStream.java:255)
    at com.example.FileProcessor.processFile(FileProcessor.java:45)
    - locked <0x000000076ad25500> (a java.io.FileInputStream)

"pool-1-thread-2" #13 prio=5 os_prio=0 tid=0x00007f8a28004000 nid=0x1238 waiting on condition [0x00007f8a37ffa000]
   java.lang.Thread.State: TIMED_WAITING (sleeping)
    at java.lang.Thread.sleep(Native Method)
    at com.example.WorkerThread.run(WorkerThread.java:20)
    - parking to wait for <0x000000076ad26000> (a java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject)

"common-pool-forkjoin-1" #14 prio=5 os_prio=0 tid=0x00007f8a28005000 nid=0x1239 runnable [0x00007f8a37ff9000]
   java.lang.Thread.State: RUNNABLE
    at java.util.concurrent.ForkJoinPool.runWorker(ForkJoinPool.java:1670)
    at java.util.concurrent.ForkJoinWorkerThread.run(ForkJoinWorkerThread.java:177)
'''
                st.session_state.raw_thread_dump_content = sample_data
                st.session_state.analysis_performed = False # Reset to allow analysis
                st.session_state.ai_analysis_result = None # Clear previous AI result
                st.rerun() # Rerun to update text area and enable analysis button
        with col_analyze_raw:
            if st.button("Analyze Raw Content", key='analyze_raw'):
                if st.session_state.raw_thread_dump_content.strip():
                    analysis_trigger = True
                    st.session_state.analysis_performed = False # Ensure re-analysis on button click
                    st.session_state.ai_analysis_result = None # Clear previous AI result
                else:
                    st.warning("Please paste thread dump content or load sample data first.")
    
    # Perform analysis if triggered and not already performed
    if analysis_trigger and not st.session_state.analysis_performed:
        all_files_content = {}
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    files_content = extract_files_from_archive(uploaded_file)
                    all_files_content.update(files_content)
            
        elif st.session_state.raw_thread_dump_content.strip():
            all_files_content = {'pasted_content.txt': st.session_state.raw_thread_dump_content}

        if all_files_content:
            st.success(f"✅ Input processed. Analyzing {len(all_files_content)} file(s).")

            with st.spinner("Analyzing thread dumps..."):
                if len(all_files_content) == 1:
                    content = list(all_files_content.values())[0]
                    st.session_state.analyzer.parse_thread_dump(content)
                else:
                    st.session_state.analyzer.parse_multiple_files(all_files_content)
            st.session_state.analysis_performed = True
            st.rerun() # Rerun to display results in tabs
        else:
            st.warning("No thread dump content provided for analysis.")

    # Display results if analysis has been performed
    if st.session_state.analysis_performed and st.session_state.analyzer.threads:
        st.divider()
        st.header("📊 Analysis Results")

        tab_summary, tab_charts, tab_deadlocks, tab_all_threads, tab_ai_analysis = st.tabs(
            ["Summary", "Charts", "Deadlocks", "All Threads", "AI Analysis"]
        )

        with tab_summary:
            st.subheader("Summary of Thread Dump Analysis")
            st.markdown(st.session_state.analyzer.summary.get('Summary_Text', 'No summary generated.'))

            # Add a download button for the Excel report here
            output_buffer = io.BytesIO()
            success = save_results_to_excel(
                st.session_state.analyzer, 
                output_buffer, 
                aianalysis_enabled, 
                st.session_state.ai_analysis_result # Pass the stored AI analysis result
            )
            if success:
                st.download_button(
                    label="Download Analysis Report (Excel)",
                    data=output_buffer.getvalue(),
                    file_name="thread_dump_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download a detailed Excel report with all parsed data and summary statistics, including AI analysis if performed."
                )

        with tab_charts:
            create_advanced_charts(st.session_state.analyzer)

        with tab_deadlocks:
            st.subheader("Detected Deadlocks")
            if st.session_state.analyzer.deadlocks:
                for i, dl in enumerate(st.session_state.analyzer.deadlocks):
                    st.error(f"⚠️ **Deadlock {i + 1} Detected!**")
                    st.markdown(f"**Thread 1:** `{dl['thread1']}` (ID: `{next((t['id'] for t in st.session_state.analyzer.threads if t['name'] == dl['thread1']), 'N/A')}`)")
                    st.markdown(f"  - Waiting for lock: `<{dl['thread1_waiting_for_lock']}>`")
                    st.markdown(f"  - Held by: `{dl['thread2']}`")
                    st.markdown(f"**Thread 2:** `{dl['thread2']}` (ID: `{next((t['id'] for t in st.session_state.analyzer.threads if t['name'] == dl['thread2']), 'N/A')}`)")
                    st.markdown(f"  - Waiting for lock: `<{dl['thread2_waiting_for_lock']}>`")
                    st.markdown(f"  - Held by: `{dl['thread1']}`")
                    st.markdown("---")
            else:
                st.success("No deadlocks detected in the provided thread dumps. 🎉")

        with tab_all_threads:
            st.subheader("All Parsed Threads")
            if st.session_state.analyzer.threads:
                threads_df = pd.DataFrame(st.session_state.analyzer.threads)
                # Drop raw_block column for display if it exists, as it can be very verbose
                display_df = threads_df.drop(columns=['raw_block'], errors='ignore')
                display_df['stack_trace'] = display_df['stack_trace'].apply(lambda x: "\n".join(x))
                display_df['locks_held'] = display_df['locks_held'].apply(lambda x: ", ".join(x))
                display_df['locks_waiting'] = display_df['locks_waiting'].apply(lambda x: ", ".join(x))

                search_query = st.text_input("Search threads by name, ID, or state:", "")
                if search_query:
                    search_query_lower = search_query.lower()
                    display_df = display_df[
                        display_df.apply(
                            lambda row: row.astype(str).str.lower().str.contains(search_query_lower, na=False).any(),
                            axis=1
                        )
                    ]

                st.dataframe(display_df, height=600, use_container_width=True)
                st.info("💡 You can click on column headers to sort the table.")
            else:
                st.info("No threads were parsed from the input.")

        with tab_ai_analysis:
            if aianalysis_enabled:
                ai_input_snippet = st.session_state.analyzer.summary.get('Summary_Text', '')
                if ai_input_snippet.strip():
                    if len(ai_input_snippet) > 8192: # Ollama context window limit
                        ai_input_snippet = ai_input_snippet[:8192] + "\n... (summary truncated for AI analysis due to length)"
                        st.warning("Summary truncated for AI analysis due to excessive length.")
                    
                    # Run AI analysis and store the result
                    st.session_state.ai_analysis_result = run_ai_analysis(
                        ai_input_snippet, ollama_api_url, ollama_model_name, prompt_template_config
                    )
                else:
                    st.info("No summary data generated to send for AI analysis. Please ensure a valid thread dump was processed.")
            else:
                st.info("AI Analysis is disabled. Enable it from the sidebar settings if needed.")

    st.markdown("---")
    st.info("For more advanced insights and large-scale analysis, consider using specialized tools like [FastThread.io](https://fastthread.io/).")


if __name__ == "__main__":
    main()