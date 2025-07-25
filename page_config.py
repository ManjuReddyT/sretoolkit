import streamlit as st

# =====================
# PAGE DEFINITIONS
# =====================

home_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

about_page = st.Page(
    "views/about.py",
    title="About",
    icon=":material/info:"
)

admin_page = st.Page(
    "views/admin.py",
    title="Admin",
    icon=":material/admin_panel_settings:",
)

reports_page = st.Page(
    "views/reports.py",
    title="Recent Reports",
    icon=":material/analytics:",
)

orders_dashboard = st.Page(
    "views/orderdashboard.py",
    title="Orders Dashboard",
    icon=":material/shopping_cart:",
)

prometheus_metrics = st.Page(
    "views/prmMetricsApp.py",
    title="Prometheus Metrics",
    icon=":material/monitor_heart:",
)

thanos_metrics = st.Page(
    "views/thanosMetricsApp.py",
    title="Thanos Metrics",
    icon=":material/storage:",
)

gcp_lb_metrics = st.Page(
    "views/gcpLBLogsVisualizer.py",
    title="GCP LB Logs",
    icon=":material/cloud:",
)

newrelic_data = st.Page(
    "views/newRelicData.py",
    title="NewRelic Data",
    icon=":material/bug_report:",
)

lighthouse_runner = st.Page(
    "runlighthouse.py",
    title="Run Lighthouse",
    icon=":material/flash_on:",
)

lighthouse_reports = st.Page(
    "views/lighthouse/lhdashboard.py",
    title="Lighthouse Reports",
    icon=":material/bar_chart:",
)

crux_reports = st.Page(
    "views/cruxDashboard.py",
    title="CRUX Reports",
    icon=":material/public:",
)

apk_metrics = st.Page(
    "views/apkMetrics.py",
    title="Android/iOS Metrics",
    icon=":material/android:",
)

run_tool = st.Page(
    "views/runpage.py",
    title="Run Tool",
    icon=":material/play_arrow:",
)

quick_guide = st.Page(
    "views/quickGuide.py",
    title="SRE Quick Guide",
    icon=":material/help_outline:",
)

jmeter_dashboard = st.Page(
    "views/JmeterDashBoard.py",
    title="JMeter Dashboard",
    icon=":material/analytics:",
)

jfr_analyzer = st.Page(
    "views/jfr_analyzer_app.py",
    title="JFR Analyzer",
    icon=":material/track_changes:",
)

gc_log_analyzer = st.Page(
    "views/gcanalyzer/gcanalyzerapp.py",
    title="GC Log Analyzer",
    icon=":material/memory:",
)

chatbot_page = st.Page(
    "views/chatbot.py",
    title="Chat with Ollama",
    icon=":material/smart_toy:",
)

chatpdfbot_page = st.Page(
    "views/chatWithPDF.py",
    title="Chat With PDF",
    icon=":material/picture_as_pdf:",
)

# =====================
# PAGE NAVIGATION STRUCTURE
# =====================

NAV_STRUCTURE = {
    "Home": [home_page],
    "SRE Tools": [
        run_tool,
        jmeter_dashboard,
        gc_log_analyzer,
        jfr_analyzer,
        prometheus_metrics,
        thanos_metrics,
        newrelic_data,
        gcp_lb_metrics,
    ],
    "UX Reports": [
        lighthouse_runner,
        lighthouse_reports,
        apk_metrics,
        crux_reports,
    ],
    "Reports": [reports_page, orders_dashboard],
    "Utilities": [quick_guide, chatbot_page, chatpdfbot_page],
    "Admin": [admin_page],
}
