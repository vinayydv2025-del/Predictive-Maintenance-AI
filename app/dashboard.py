
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "platform_integration_report.json"
)

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
    }

    [data-testid="stSidebar"] {
        background-color: #111c30;
    }

    .hero {
        padding: 24px 28px;
        border: 1px solid #253650;
        border-radius: 14px;
        background: linear-gradient(120deg, #14243c, #101a2d);
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #f1f5f9;
        font-size: 29px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #a9bad0;
        font-size: 14px;
        margin-bottom: 0;
    }

    .section-heading {
        color: #e2e8f0;
        font-size: 21px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #111c30;
        border: 1px solid #253650;
        padding: 15px;
        border-radius: 11px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bad0;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #253650;
        border-radius: 9px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False, ttl=30)
def load_report(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if not REPORT_PATH.exists():
    st.error(
        "Unified platform report was not found."
    )
    st.code("python run_platform.py")
    st.stop()

try:
    report = load_report(str(REPORT_PATH))
except (json.JSONDecodeError, OSError) as error:
    st.error(f"Could not load platform report: {error}")
    st.stop()

if not isinstance(report, dict):
    st.error("Invalid platform report format.")
    st.stop()


# ============================================================
# SHARED DATA
# ============================================================

scenario_report = report.get(
    "scenario_fleet_intelligence", {}
)

maintenance_plan = report.get(
    "maintenance_plan", {}
)

optimization_report = report.get(
    "optimization_results", {}
)

fleet_analytics = scenario_report.get(
    "fleet_analytics", {}
)

fleet_health = fleet_analytics.get(
    "fleet_health", {}
)

fleet_overview = fleet_health.get(
    "fleet_overview", {}
)

scenario_results = scenario_report.get(
    "scenario_results", []
)

work_orders = maintenance_plan.get(
    "work_orders", []
)

experiments = optimization_report.get(
    "scenario_results", []
)

aggregates = optimization_report.get(
    "aggregated_results", {}
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_mapping(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def display_number(value, decimals=2):
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}"
    return "—"


def display_percent(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return "—"


def count_dataframe(mapping, category_name="Category"):
    mapping = safe_mapping(mapping)

    rows = []

    for key, value in mapping.items():
        if isinstance(value, (int, float)):
            rows.append({
                category_name: str(key),
                "Count": value,
            })

    return pd.DataFrame(rows)


def render_count_chart(mapping, title, chart_type="bar"):
    dataframe = count_dataframe(mapping)

    if dataframe.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    if chart_type == "pie":
        figure = px.pie(
            dataframe,
            names="Category",
            values="Count",
            hole=0.55,
            title=title,
        )
    else:
        figure = px.bar(
            dataframe,
            x="Category",
            y="Count",
            title=title,
            text_auto=True,
        )

    figure.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=15, r=15, t=55, b=15),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_table(data, preferred_columns=None):
    if not data:
        st.info("No records available.")
        return

    dataframe = pd.DataFrame(data)

    if preferred_columns:
        available = [
            column
            for column in preferred_columns
            if column in dataframe.columns
        ]

        if available:
            dataframe = dataframe[available]

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )


def get_machine_label(item):
    machine_id = item.get("machine_id", "Unknown")
    scenario = item.get("scenario", "Unknown")
    severity = item.get("severity", "Unknown")

    return (
        f"{machine_id} | {scenario} | {severity}"
    )


def count_solver_statuses(experiment_data):
    statuses = {}

    for scenario in safe_list(experiment_data):
        for result in safe_list(
            scenario.get("results", [])
        ):
            if result.get("strategy") == "OR_TOOLS_OPTIMAL":
                status = result.get(
                    "solver_status",
                    "UNKNOWN",
                )

                statuses[status] = (
                    statuses.get(status, 0) + 1
                )

    return statuses


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏭 Maintenance AI")

st.sidebar.caption(
    "Industrial Predictive Maintenance Platform"
)

st.sidebar.divider()

pages = [
    "Executive Overview",
    "Fleet Intelligence",
    "AI Diagnostics",
    "Maintenance Center",
    "Optimization Lab",
]

selected_page = st.sidebar.radio(
    "Navigation",
    pages,
)

if st.sidebar.button(
    "↻ Refresh latest report",
    width="stretch",
):
    load_report.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    f"Report generated: {report.get('generated_at', 'Unknown')}"
)

st.sidebar.caption(
    "Data source: Synthetic Industrial Digital Twin"
)

st.sidebar.caption(
    "Platform version: "
    f"{report.get('platform_version', 'Unknown')}"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Industrial Predictive Maintenance & Intelligence</h1>
        <p>
            Digital Twin • Hybrid AI Diagnostics • Fleet Intelligence
            • Maintenance Planning • CP-SAT Optimization
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if selected_page == "Executive Overview":

    st.markdown(
        '<div class="section-heading">Platform Overview</div>',
        unsafe_allow_html=True,
    )

    summary = safe_mapping(
        report.get("platform_summary")
    )

    maintenance_summary = safe_mapping(
        maintenance_plan.get(
            "fleet_maintenance_summary"
        )
    )

    solver_statuses = count_solver_statuses(
        experiments
    )

    total_machines = summary.get(
        "machines_simulated",
        len(scenario_results),
    )

    work_order_count = summary.get(
        "work_orders_generated",
        len(work_orders),
    )

    avg_physical_health = safe_mapping(
        scenario_report.get("simulator_status")
    ).get("average_health")

    diagnostic_health = fleet_overview.get(
        "fleet_health_percentage"
    )

    fleet_status = fleet_overview.get(
        "fleet_status",
        "NOT AVAILABLE",
    )

    estimated_cost = summary.get(
        "estimated_maintenance_cost_simulation_units"
    )

    downtime = maintenance_summary.get(
        "total_estimated_downtime_hours"
    )

    optimality_count = summary.get(
        "optimality_proven_count",
        0,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Machines Simulated",
        total_machines,
    )

    col2.metric(
        "Fleet Classification",
        fleet_status,
    )

    col3.metric(
        "Maintenance Work Orders",
        work_order_count,
    )

    col4.metric(
        "Optimization Scenarios",
        summary.get("optimization_scenarios", 0),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Physical Machine Health",
        display_percent(avg_physical_health),
        help="Average physical health reported by the simulator.",
    )

    col2.metric(
        "Machines Classified NORMAL",
        display_percent(diagnostic_health),
        help=(
            "Percentage classified as NORMAL by fleet diagnostics. "
            "This is not average physical machine health."
        ),
    )

    col3.metric(
        "Estimated Maintenance Cost",
        display_number(estimated_cost, 0),
        help="Simulation units, not validated real-world currency.",
    )

    col4.metric(
        "Estimated Downtime",
        f"{display_number(downtime)} hours",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        render_count_chart(
            fleet_health.get(
                "severity_distribution", {}
            ),
            "Fleet Severity Distribution",
            "pie",
        )

    with right:
        render_count_chart(
            fleet_health.get(
                "diagnosis_distribution", {}
            ),
            "Diagnosis Distribution",
            "bar",
        )

    left, right = st.columns(2)

    with left:
        render_count_chart(
            fleet_analytics.get(
                "fault_frequency", {}
            ),
            "Detected Fault Frequency",
            "bar",
        )

    with right:
        render_count_chart(
            fleet_analytics.get(
                "maintenance_action_distribution", {}
            ),
            "Maintenance Action Distribution",
            "pie",
        )

    st.divider()

    st.subheader("Highest Priority Machine")

    highest_priority = safe_mapping(
        fleet_analytics.get(
            "highest_priority_machine"
        )
    )

    if highest_priority:
        a, b, c = st.columns(3)

        a.metric(
            "Machine",
            highest_priority.get(
                "machine_id",
                "Unknown",
            ),
        )

        b.metric(
            "Severity",
            highest_priority.get(
                "severity",
                "Unknown",
            ),
        )

        c.metric(
            "Risk Score",
            display_number(
                highest_priority.get(
                    "risk_score",
                    highest_priority.get("score"),
                )
            ),
        )

    else:
        st.info("No highest-priority machine is available.")

    st.subheader("Optimization Solver Status")

    status_dataframe = count_dataframe(
        solver_statuses,
        "Solver Status",
    )

    if not status_dataframe.empty:
        st.dataframe(
            status_dataframe,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Solver optimality is established only for scenarios "
        "reported with status OPTIMAL."
    )


# ============================================================
# PAGE 2: FLEET INTELLIGENCE
# ============================================================

elif selected_page == "Fleet Intelligence":

    st.markdown(
        '<div class="section-heading">Fleet Intelligence Center</div>',
        unsafe_allow_html=True,
    )

    critical_count = fleet_analytics.get(
        "critical_machine_count",
        0,
    )

    high_risk_count = fleet_analytics.get(
        "high_risk_machine_count",
        0,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Critical Machines",
        critical_count,
    )

    col2.metric(
        "High Risk Machines",
        high_risk_count,
    )

    col3.metric(
        "Total Machines Analyzed",
        fleet_analytics.get(
            "analytics_metadata", {}
        ).get("total_machines_analyzed", len(scenario_results)),
    )

    st.divider()

    st.subheader("Machine Risk Ranking")

    risk_ranking = safe_list(
        fleet_analytics.get("risk_ranking", [])
    )

    if risk_ranking:
        ranking_dataframe = pd.json_normalize(
            risk_ranking
        )

        st.dataframe(
            ranking_dataframe,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Ranking is rule-based using severity and evidence. "
            "Risk scores are not calibrated probabilities."
        )
    else:
        st.info("No risk ranking data available.")

    st.divider()

    st.subheader("Machine-Level Inspection")

    if scenario_results:
        machine_index = st.selectbox(
            "Select a machine",
            options=list(range(len(scenario_results))),
            format_func=lambda index: get_machine_label(
                scenario_results[index]
            ),
        )

        machine = scenario_results[machine_index]

        diagnosis = safe_mapping(
            machine.get("final_diagnosis")
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Machine ID",
            machine.get("machine_id", "Unknown"),
        )

        col2.metric(
            "Diagnosis",
            diagnosis.get("diagnosis", "Unknown"),
        )

        col3.metric(
            "Severity",
            diagnosis.get("severity", "Unknown"),
        )

        col4.metric(
            "Diagnostic Reliability",
            diagnosis.get(
                "diagnostic_reliability",
                "Unknown",
            ),
        )

        st.write(
            "**Maintenance action:**",
            diagnosis.get("maintenance_action", "Not available"),
        )

        st.write(
            "**Detected faults:**",
            ", ".join(
                machine.get("detected_faults", [])
            ) or "None",
        )

        st.write(
            "**Recommendations:**"
        )

        recommendations = diagnosis.get(
            "recommendations", []
        )

        if recommendations:
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
        else:
            st.info("No recommendations available.")

        with st.expander("Sensor Quality Details"):
            st.json(
                safe_mapping(
                    diagnosis.get("sensor_quality")
                )
            )

    st.divider()

    left, right = st.columns(2)

    with left:
        render_count_chart(
            fleet_analytics.get(
                "maintenance_action_distribution", {}
            ),
            "Maintenance Action Distribution",
            "bar",
        )

    with right:
        render_count_chart(
            fleet_analytics.get(
                "fault_frequency", {}
            ),
            "Fault Frequency",
            "bar",
        )

    with st.expander("Sensor Quality Fleet Summary"):
        st.json(
            fleet_analytics.get(
                "sensor_quality_summary", {}
            )
        )


# ============================================================
# PAGE 3: AI DIAGNOSTICS
# ============================================================

elif selected_page == "AI Diagnostics":

    st.markdown(
        '<div class="section-heading">Hybrid AI Diagnostics</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Compare synthetic scenario fault injection with "
        "faults detected by the diagnostic engine."
    )

    comparison_rows = []

    for item in scenario_results:
        injected = set(
            item.get("injected_faults", [])
        )

        detected = set(
            item.get("detected_faults", [])
        )

        missed = sorted(injected - detected)
        additional = sorted(detected - injected)

        if not missed and not additional:
            comparison = "Exact label agreement"
        elif detected and not missed:
            comparison = "Injected labels detected; extra labels present"
        elif detected:
            comparison = "Partial or mismatched detection"
        else:
            comparison = "No injected labels detected"

        comparison_rows.append({
            "Machine": item.get("machine_id"),
            "Scenario": item.get("scenario"),
            "Injected Faults": ", ".join(sorted(injected)) or "None",
            "Detected Faults": ", ".join(sorted(detected)) or "None",
            "Missed Injected Labels": ", ".join(missed) or "None",
            "Additional Detected Labels": ", ".join(additional) or "None",
            "Sensor Issue Injected": item.get(
                "injected_sensor_quality_issue",
                False,
            ),
            "Label Comparison": comparison,
        })

    st.subheader("Scenario Diagnostic Comparison")

    comparison_dataframe = pd.DataFrame(
        comparison_rows
    )

    if not comparison_dataframe.empty:
        st.dataframe(
            comparison_dataframe,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "This comparison uses controlled synthetic fault scenarios. "
        "It does not establish real-world model accuracy or "
        "industrial generalization."
    )

    st.divider()

    st.subheader("Diagnostic Deep Dive")

    if scenario_results:
        selected_index = st.selectbox(
            "Select diagnostic record",
            options=list(range(len(scenario_results))),
            format_func=lambda index: get_machine_label(
                scenario_results[index]
            ),
            key="diagnostic_machine",
        )

        selected_machine = scenario_results[
            selected_index
        ]

        diagnostic_result = safe_mapping(
            selected_machine.get("diagnostic_result")
        )

        final_diagnosis = safe_mapping(
            diagnostic_result.get(
                "final_diagnosis",
                selected_machine.get("final_diagnosis", {}),
            )
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Final Diagnosis",
            final_diagnosis.get(
                "diagnosis",
                "Unknown",
            ),
        )

        col2.metric(
            "Severity",
            final_diagnosis.get(
                "severity",
                "Unknown",
            ),
        )

        col3.metric(
            "Supporting Alert Count",
            final_diagnosis.get(
                "supporting_alert_count",
                0,
            ),
        )

        st.write(
            "**Diagnostic reliability:**",
            final_diagnosis.get(
                "diagnostic_reliability",
                "Unknown",
            ),
        )

        st.write(
            "**Recommended maintenance action:**",
            final_diagnosis.get(
                "maintenance_action",
                "Unknown",
            ),
        )

        with st.expander("Final Diagnosis Details", expanded=True):
            st.json(final_diagnosis)

        model_outputs = safe_mapping(
            diagnostic_result.get("model_outputs")
        )

        with st.expander("Individual AI Model Outputs"):
            if model_outputs:
                for model_name, model_result in model_outputs.items():
                    st.markdown(f"**{model_name}**")
                    st.json(model_result)
            else:
                st.info("Individual model outputs are unavailable.")

        with st.expander("Sensor Quality Assessment"):
            st.json(
                final_diagnosis.get(
                    "sensor_quality",
                    {},
                )
            )

        with st.expander("Diagnostic Limitations"):
            limitations = safe_list(
                diagnostic_result.get("limitations")
            )

            if limitations:
                for limitation in limitations:
                    st.write(f"- {limitation}")
            else:
                st.info("No limitation metadata found.")


# ============================================================
# PAGE 4: MAINTENANCE CENTER
# ============================================================

elif selected_page == "Maintenance Center":

    st.markdown(
        '<div class="section-heading">Maintenance Planning Center</div>',
        unsafe_allow_html=True,
    )

    summary = safe_mapping(
        maintenance_plan.get(
            "fleet_maintenance_summary"
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Work Orders",
        summary.get("work_orders_created", len(work_orders)),
    )

    col2.metric(
        "Schedule Days",
        summary.get("schedule_days_required", 0),
    )

    col3.metric(
        "Estimated Downtime",
        f"{display_number(summary.get('total_estimated_downtime_hours'))} hrs",
    )

    col4.metric(
        "Estimated Cost",
        display_number(
            summary.get(
                "total_estimated_cost_simulation_units"
            ),
            0,
        ),
        help="Simulation units, not validated real-world currency.",
    )

    st.divider()

    st.subheader("Maintenance Work Orders")

    preferred_columns = [
        "machine_id",
        "severity",
        "diagnosis",
        "work_order_type",
        "work_order_status",
        "priority_score",
        "scheduled_day",
        "assigned_technician",
        "estimated_duration_hours",
        "estimated_total_cost",
        "estimated_downtime_hours",
        "equipment_repair_authorized",
        "diagnostic_reliability",
        "recommended_action",
        "detected_faults",
    ]

    render_table(
        work_orders,
        preferred_columns,
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Spare Parts Forecast")

        parts_forecast = safe_list(
            maintenance_plan.get(
                "spare_parts_forecast", []
            )
        )

        render_table(
            parts_forecast,
            [
                "part_name",
                "required_quantity",
                "planning_basis",
            ],
        )

    with right:
        st.subheader("Daily Technician Capacity")

        capacity = safe_list(
            maintenance_plan.get(
                "daily_technician_capacity", []
            )
        )

        capacity_rows = []

        for day in capacity:
            hours = safe_list(
                day.get("technician_hours", [])
            )

            for index, technician_hours in enumerate(hours):
                capacity_rows.append({
                    "Day": day.get("day"),
                    "Technician": f"TECHNICIAN-{index + 1:02d}",
                    "Allocated Hours": technician_hours,
                })

        if capacity_rows:
            capacity_dataframe = pd.DataFrame(
                capacity_rows
            )

            figure = px.bar(
                capacity_dataframe,
                x="Day",
                y="Allocated Hours",
                color="Technician",
                barmode="group",
                title="Technician Hours by Day",
                text_auto=True,
            )

            figure.update_layout(
                template="plotly_dark",
                height=330,
            )

            st.plotly_chart(
                figure,
                width="stretch",
            )
        else:
            st.info("No technician capacity data available.")

    st.divider()

    with st.expander("Planning Assumptions"):
        st.json(
            maintenance_plan.get(
                "planning_assumptions", {}
            )
        )

    with st.expander("Maintenance Validation"):
        st.json(
            maintenance_plan.get(
                "validation", {}
            )
        )


# ============================================================
# PAGE 5: OPTIMIZATION LAB
# ============================================================

elif selected_page == "Optimization Lab":

    st.markdown(
        '<div class="section-heading">Maintenance Optimization Lab</div>',
        unsafe_allow_html=True,
    )

    recommendation = optimization_report.get(
        "recommended_strategy",
        "Unknown",
    )

    strategy_aggregates = safe_mapping(
        aggregates
    )

    cp_metrics = safe_mapping(
        strategy_aggregates.get(
            "OR_TOOLS_OPTIMAL"
        )
    )

    cp_solver_statuses = count_solver_statuses(
        experiments
    )

    total_solver_runs = sum(
        cp_solver_statuses.values()
    )

    optimal_runs = cp_solver_statuses.get(
        "OPTIMAL",
        0,
    )

    optimality_rate = (
        optimal_runs / total_solver_runs * 100
        if total_solver_runs > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Optimization Scenarios",
        len(experiments),
    )

    col2.metric(
        "Recommended Strategy",
        recommendation,
    )

    col3.metric(
        "CP-SAT Average Makespan",
        f"{display_number(cp_metrics.get('average_makespan_hours'))} hrs",
    )

    col4.metric(
        "Optimality Proven",
        f"{optimal_runs}/{total_solver_runs}",
        display_percent(optimality_rate),
    )

    st.divider()

    st.subheader("Scheduling Strategy Comparison")

    if strategy_aggregates:
        aggregate_dataframe = (
            pd.DataFrame.from_dict(
                strategy_aggregates,
                orient="index",
            )
            .rename_axis("Strategy")
            .reset_index()
        )

        st.dataframe(
            aggregate_dataframe,
            width="stretch",
            hide_index=True,
        )

        metric_columns = [
            column
            for column in [
                "average_makespan_hours",
                "average_critical_completion_hours",
            ]
            if column in aggregate_dataframe.columns
        ]

        if metric_columns:
            chart_dataframe = aggregate_dataframe.melt(
                id_vars=["Strategy"],
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Hours",
            )

            figure = px.bar(
                chart_dataframe,
                x="Strategy",
                y="Hours",
                color="Metric",
                barmode="group",
                title="Average Scheduling Performance",
                text_auto=".2f",
            )

            figure.update_layout(
                template="plotly_dark",
                height=420,
            )

            st.plotly_chart(
                figure,
                width="stretch",
            )

    st.divider()

    st.subheader("Scenario Win Counts")

    win_counts = safe_mapping(
        optimization_report.get(
            "scenario_win_counts"
        )
    )

    render_count_chart(
        win_counts,
        "Number of Scenario Wins",
        "bar",
    )

    st.divider()

    st.subheader("CP-SAT Scenario Inspection")

    if experiments:
        selected_scenario_index = st.selectbox(
            "Select optimization scenario",
            options=list(range(len(experiments))),
            format_func=lambda index: (
                f"{experiments[index].get('scenario_id', index)}"
                f" | {experiments[index].get('scenario_type', 'Unknown')}"
            ),
        )

        selected_scenario = experiments[
            selected_scenario_index
        ]

        cp_result = next(
            (
                result
                for result in selected_scenario.get("results", [])
                if result.get("strategy") == "OR_TOOLS_OPTIMAL"
            ),
            None,
        )

        if cp_result:
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Solver Status",
                cp_result.get(
                    "solver_status",
                    "Unknown",
                ),
            )

            col2.metric(
                "Optimality Proven",
                "Yes" if cp_result.get(
                    "optimality_proven",
                    False,
                ) else "No",
            )

            col3.metric(
                "Solver Runtime",
                f"{display_number(cp_result.get('solver_wall_time_seconds'), 4)} sec",
            )

            col4.metric(
                "Makespan",
                f"{display_number(cp_result.get('makespan_hours'))} hrs",
            )

            schedule = safe_list(
                cp_result.get(
                    "scheduled_work_orders", []
                )
            )

            st.subheader("Optimized Work Schedule")

            render_table(
                schedule,
                [
                    "machine_id",
                    "severity",
                    "diagnosis",
                    "assigned_technician",
                    "start_hour",
                    "finish_hour",
                    "waiting_hours",
                    "duration",
                ],
            )

            with st.expander("Solver Details"):
                st.json({
                    "solver_status": cp_result.get(
                        "solver_status"
                    ),
                    "solver_objective_value": cp_result.get(
                        "solver_objective_value"
                    ),
                    "solver_best_bound": cp_result.get(
                        "solver_best_bound"
                    ),
                    "optimality_proven": cp_result.get(
                        "optimality_proven"
                    ),
                    "warm_start_enabled": cp_result.get(
                        "warm_start_enabled"
                    ),
                    "warm_start_strategy": cp_result.get(
                        "warm_start_strategy"
                    ),
                })

        else:
            st.warning(
                "No CP-SAT result found for this scenario."
            )

    st.divider()

    with st.expander("Optimization Methodology"):
        st.write(
            optimization_report.get(
                "objective",
                "Objective not available.",
            )
        )

        st.write(
            optimization_report.get(
                "recommendation_basis",
                "Recommendation basis not available.",
            )
        )

        st.json(
            optimization_report.get(
                "scenario_generation",
                {},
            )
        )

        st.json(
            optimization_report.get(
                "warm_start",
                {},
            )
        )

    with st.expander("Optimization Validation"):
        st.json(
            optimization_report.get(
                "validation",
                {},
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Predictive Maintenance & Intelligence Platform "
    "| Research Prototype"
)

st.caption(
    "Synthetic simulation data. AI model outputs, risk rankings, "
    "maintenance costs and scheduling results require further "
    "real-world validation. This is not a certified industrial "
    "safety system."
)
