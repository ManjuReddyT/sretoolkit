import streamlit as st

def render_key_concepts():
    """Renders the Key SRE Concepts tab."""
    st.header("Key SRE Concepts")
    st.markdown("""
        Understanding these core principles is fundamental to Site Reliability Engineering. They help in quantifying and managing the reliability of your services.
    """)

    with st.expander("Service Level Indicators (SLIs)", expanded=True):
        st.markdown("""
            **What it is:** A quantitative measure of some aspect of the level of service that is provided.

            An SLI is a metric that tells you how well your service is performing. It's the foundation of your SLOs and error budgets.

            - **Examples:**
                - Request latency (e.g., percentage of requests served in under 100ms)
                - Error rate (e.g., percentage of failed requests)
                - System throughput (e.g., requests per second)
                - Availability (e.g., fraction of time the service is usable)
        """)
        st.code("SLI: (Good Events / Valid Events) * 100", language='text')

    with st.expander("Service Level Objectives (SLOs)"):
        st.markdown("""
            **What it is:** A target value or range of values for a service level that is measured by an SLI.

            An SLO is a goal for your service's reliability. It's a formal agreement on the desired performance level.

            - **Examples:**
                - `99.9%` of requests will be served in under 100ms.
                - `99.99%` of API calls will be successful.
                - `99.5%` uptime over a 30-day window.
        """)
        st.code("SLO: SLI <= Target", language='text')

    with st.expander("Error Budgets"):
        st.markdown("""
            **What it is:** The amount of "unreliability" you are willing to tolerate.

            The error budget is derived from your SLO. If your SLO is 99.9% availability, your error budget is 0.1% of the time. This budget can be "spent" on deployments, maintenance, or unavoidable failures.

            - **Calculation:** `100% - SLO %`
            - **Example:** For a 99.9% SLO over a 30-day period (approx. 43,200 minutes), the error budget is:
        """)
        st.code("0.1% of 43,200 minutes = 43.2 minutes of acceptable downtime", language='text')

    with st.expander("MTTR & MTBF"):
        st.markdown("""
            **MTTR (Mean Time to Repair):** The average time it takes to repair a failed system. This includes detection, diagnosis, and resolution. A lower MTTR is better.

            **MTBF (Mean Time Between Failures):** The average time a system operates between failures. A higher MTBF is better.
        """)


def render_on_call_checklist():
    """Renders the On-Call Checklist tab."""
    st.header("On-Call Incident Response Checklist")
    st.markdown("""
        A structured approach to handling incidents when you are on-call.
    """)

    st.subheader("1. Initial Alert & Triage")
    st.markdown("""
        - **Acknowledge:** Immediately acknowledge the alert in the alerting tool (e.g., PagerDuty, Opsgenie).
        - **Assess Impact:** Quickly determine the user-facing impact. Is it a critical outage or a minor degradation?
        - **Open Communication Channel:** Create a dedicated channel for the incident (e.g., a Slack channel `#incident-YYYY-MM-DD-service`).
    """)

    st.subheader("2. Investigation & Diagnosis")
    st.markdown("""
        - **Check Dashboards:** Review the primary service dashboards (Grafana, Datadog) for anomalies.
        - **Check Logs:** Search centralized logs (ELK, Splunk) for errors or unusual patterns.
        - **Recent Changes:** Was there a recent deployment, configuration change, or feature flag rollout?
        - **Formulate Hypothesis:** Based on the data, what is the likely cause?
    """)

    st.subheader("3. Mitigation & Resolution")
    st.markdown("""
        - **Mitigate First:** The priority is to restore service. This might mean a temporary fix.
            - *Can you roll back the recent change?*
            - *Can you restart the service?*
            - *Can you fail over to a replica?*
        - **Escalate if Needed:** If you are stuck for more than 15-20 minutes, don't hesitate to escalate to a secondary on-call or subject matter expert.
        - **Apply Fix:** Once the immediate issue is resolved, apply a more permanent fix if necessary.
    """)

    st.subheader("4. Communication")
    st.markdown("""
        - **Keep Stakeholders Updated:** Post regular, concise updates in the incident channel.
        - **Update Status Page:** If you have a public status page, keep it updated for users.
    """)

    st.subheader("5. Post-Incident")
    st.markdown("""
        - **Ensure Monitoring is Green:** Verify that all systems are back to normal.
        - **Schedule Postmortem:** A blameless postmortem is crucial for learning.
        - **Handover:** If your on-call shift is ending, provide a clear handover to the next person.
    """)


def render_tools_and_commands():
    """Renders the Tools & Commands tab."""
    st.header("Common Tools & Commands")

    st.subheader("Linux / General")
    with st.expander("System & Process Inspection"):
        st.code("df -h  # Check disk space\n"
                "free -m # Check free memory\n"
                "top     # Monitor processes and system stats\n"
                "ps aux | grep [process_name] # Find a process\n"
                "netstat -tulnp # List listening ports", language='bash')

    with st.expander("Log Analysis"):
        st.code("tail -f /var/log/service.log # Follow a log file\n"
                "grep 'ERROR' /var/log/service.log # Search for errors\n"
                "journalctl -u [service_name] -n 100 # View systemd logs", language='bash')

    st.subheader("Kubernetes (`kubectl`)")
    with st.expander("Pod & Node Management"):
        st.code("kubectl get pods -n [namespace] # List pods\n"
                "kubectl describe pod [pod_name] -n [namespace] # Get pod details\n"
                "kubectl logs -f [pod_name] -n [namespace] # Follow pod logs\n"
                "kubectl exec -it [pod_name] -n [namespace] -- /bin/bash # Shell into a pod\n"
                "kubectl get nodes -o wide # Check node status", language='bash')

    st.subheader("Docker")
    with st.expander("Container & Image Management"):
        st.code("docker ps # List running containers\n"
                "docker logs -f [container_id] # Follow container logs\n"
                "docker exec -it [container_id] /bin/bash # Shell into a container\n"
                "docker inspect [container_id] # Get container details\n"
                "docker images # List local images", language='bash')

    st.subheader("Networking (`openssl`, `curl`)")
    with st.expander("Connectivity & Certificates"):
        st.code("curl -v http://[service_url] # Verbose request to a service\n"
                "openssl s_client -connect [host]:443 # Check SSL certificate", language='bash')


def render_troubleshooting_guide():
    """Renders the Troubleshooting Guide tab."""
    st.header("Basic Troubleshooting Guide")

    st.subheader("Service is Unresponsive / High Latency")
    st.markdown("""
        1. **Check CPU/Memory:** Are the service's pods/VMs resource-constrained? Look at CPU utilization and memory usage dashboards.
        2. **Check for Errors:** Look at the service logs for a sudden spike in errors or exceptions.
        3. **Database Health:** Is the database connected to the service healthy? Check its CPU, memory, and query performance.
        4. **Network Connectivity:** Is there a networking issue between services? Check for DNS resolution problems or firewall blocks.
        5. **Recent Deployments:** Correlate the issue with the timeline of recent code or infrastructure changes.
    """)

    st.subheader("High Error Rate")
    st.markdown("""
        1. **Analyze Logs:** Grep for `ERROR`, `FATAL`, `Exception`. What is the specific error message?
        2. **Check Dependencies:** Is an upstream or downstream service failing?
        3. **Configuration:** Was there a recent configuration change (e.g., in a ConfigMap, environment variable, or feature flag)?
        4. **Bad Input:** Is a specific type of request or user input causing the failure?
    """)


def render_useful_links():
    """Renders the Useful Links tab."""
    st.header("Useful Links")
    st.markdown("A central place for important SRE-related links. **Customize these for your organization.**")

    links = {
        "Dashboards": {
            "Grafana": "http://grafana.your-company.com",
            "Datadog": "http://datadog.your-company.com",
            "Kibana": "http://kibana.your-company.com",
        },
        "Alerting & On-Call": {
            "PagerDuty": "https://your-company.pagerduty.com",
            "Opsgenie": "https://your-company.opsgenie.com",
        },
        "Documentation": {
            "Runbooks": "https://confluence.your-company.com/display/SRE/Runbooks",
            "Postmortems": "https://confluence.your-company.com/display/SRE/Postmortems",
            "Architecture Diagrams": "https://confluence.your-company.com/display/SRE/Architecture",
        }
    }

    for category, category_links in links.items():
        st.subheader(category)
        for name, url in category_links.items():
            st.markdown(f"- [{name}]({url})")

# Main app rendering
st.header("SRE Quick Reference Guide")
st.markdown("Your daily dashboard for SRE principles, checklists, and commands.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Key Concepts",
    "On-Call Checklist",
    "Tools & Commands",
    "Troubleshooting Guide",
    "Useful Links"
])

with tab1:
    render_key_concepts()

with tab2:
    render_on_call_checklist()

with tab3:
    render_tools_and_commands()

with tab4:
    render_troubleshooting_guide()

with tab5:
    render_useful_links()
