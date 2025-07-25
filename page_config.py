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


crux_reports = st.Page(
    "views/cruxDashboard.py",
    title="CRUX Reports",
    icon=":material/public:",
)


run_tool = st.Page(
    "views/run_page.py",
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

thanos_metrics = st.Page(
    "views/thanos_metrics_app.py",
    title="Thanos Metrics",
    icon=":material/analytics:",
)

newrelic_data = st.Page(
    "views/newrelic_data_app.py",
    title="New Relic Data",
    icon=":material/analytics:",
)

gcp_lb_metrics = st.Page(
    "views/gcp_lb_metrics_app.py",
    title="GCP LB Metrics",
    icon=":material/analytics:",
)

lighthouse_runner = st.Page(
    "views/lighthouse_runner_app.py",
    title="Lighthouse Runner",
    icon=":material/analytics:",
)

lighthouse_reports = st.Page(
    "views/lighthouse_reports_app.py",
    title="Lighthouse Reports",
    icon=":material/analytics:",
)

apk_metrics = st.Page(
    "views/apk_metrics_app.py",
    title="APK Metrics",
    icon=":material/analytics:",
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
