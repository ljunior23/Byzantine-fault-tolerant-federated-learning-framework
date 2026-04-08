import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time

st.set_page_config(
    page_title="BFT-FL Dashboard",
    layout="wide",
)

# Styling 
st.markdown("""
<style>
    .metric-card { background: #1e2130; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .stMetric label { color: #aaa; font-size: 0.85rem; }
    h1 { color: #4fc3f7; }
    h2 { color: #81d4fa; }
</style>
""", unsafe_allow_html=True)

# Header 
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Byzantine-Fault-Tolerant Federated Learning")
    st.caption("University of Michigan-Dearborn | Trustworthy AI")
with col2:
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# Data Loading 
results_path = Path("results/all_results.json")


@st.cache_data(ttl=10)
def load_results():
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


results = load_results()

if not results:
    st.warning("No results found. Run `python experiments/run_experiments.py --quick` first.")
    st.code("python experiments/run_experiments.py --dataset mnist --rounds 20 --quick")
    st.stop()

# Build DataFrame 
rows = []
for r in results:
    cfg = r["config"]
    rows.append({
        "Strategy": cfg["strategy"],
        "Attack": cfg["attack_type"].replace("AttackType.", ""),
        "Malicious Ratio": cfg["malicious_ratio"],
        "IID": cfg["iid"],
        "Accuracy (%)": r["final_accuracy"],
        "Loss": r["final_loss"],
        "Rounds": cfg["rounds"],
        "Dataset": cfg["dataset"],
    })

df = pd.DataFrame(rows)

# Filters
st.sidebar.header("Filters")
selected_attacks = st.sidebar.multiselect("Attack Type", df["Attack"].unique(), default=list(df["Attack"].unique()))
selected_strategies = st.sidebar.multiselect("Strategy", df["Strategy"].unique(), default=list(df["Strategy"].unique()))
selected_ratios = st.sidebar.multiselect("Malicious Ratio", sorted(df["Malicious Ratio"].unique()), default=list(df["Malicious Ratio"].unique()))

filtered_df = df[
    df["Attack"].isin(selected_attacks) &
    df["Strategy"].isin(selected_strategies) &
    df["Malicious Ratio"].isin(selected_ratios)
]

# Top KPIs 
st.subheader("Performance Summary")
k1, k2, k3, k4 = st.columns(4)
with k1:
    best = filtered_df.loc[filtered_df["Accuracy (%)"].idxmax()]
    st.metric("Best Accuracy", f"{best['Accuracy (%)']:.1f}%", f"{best['Strategy']} / {best['Attack']}")
with k2:
    fedavg_df = filtered_df[filtered_df["Strategy"] == "fedavg"]
    fedavg_acc = fedavg_df["Accuracy (%)"].mean() if len(fedavg_df) else 0
    st.metric("FedAvg Avg Accuracy", f"{fedavg_acc:.1f}%")
with k3:
    adaptive_df = filtered_df[filtered_df["Strategy"] == "adaptive"]
    adaptive_acc = adaptive_df["Accuracy (%)"].mean() if len(adaptive_df) else 0
    delta = adaptive_acc - fedavg_acc
    st.metric("Adaptive FL Avg Accuracy", f"{adaptive_acc:.1f}%", f"{delta:+.1f}% vs FedAvg")
with k4:
    st.metric("Experiments Shown", len(filtered_df), f"of {len(df)} total")

# Heatmap: Accuracy by Strategy × Attack 
st.subheader("Accuracy Heatmap (Strategy vs Attack Type)")
pivot = filtered_df.groupby(["Strategy", "Attack"])["Accuracy (%)"].mean().reset_index()
pivot_table = pivot.pivot(index="Strategy", columns="Attack", values="Accuracy (%)")

fig_heat = px.imshow(
    pivot_table,
    text_auto=".1f",
    color_continuous_scale="RdYlGn",
    zmin=0, zmax=100,
    title="Mean Test Accuracy (%) by Defense × Attack",
    aspect="auto",
)
fig_heat.update_layout(height=400, template="plotly_dark")
st.plotly_chart(fig_heat, use_container_width=True)

# Line Charts 
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Accuracy vs Malicious Ratio")
    ratio_df = filtered_df.groupby(["Strategy", "Malicious Ratio"])["Accuracy (%)"].mean().reset_index()
    fig_line = px.line(
        ratio_df, x="Malicious Ratio", y="Accuracy (%)", color="Strategy",
        markers=True, template="plotly_dark",
        title="Accuracy Degradation under Increasing Attack Intensity"
    )
    fig_line.add_hline(y=85, line_dash="dash", line_color="orange",
                       annotation_text="85% target", annotation_position="left")
    st.plotly_chart(fig_line, use_container_width=True)

with col_b:
    st.subheader("Strategy Comparison by Attack Type")
    fig_bar = px.bar(
        filtered_df.groupby(["Strategy", "Attack"])["Accuracy (%)"].mean().reset_index(),
        x="Attack", y="Accuracy (%)", color="Strategy",
        barmode="group", template="plotly_dark",
        title="Defense Strategy Performance per Attack Type"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Training History 
st.subheader("Training Curves")
history_rows = []
for r in results:
    cfg = r["config"]
    if cfg["strategy"] not in selected_strategies:
        continue
    if cfg["attack_type"].replace("AttackType.", "") not in selected_attacks:
        continue
    for h in r.get("accuracy_history", []):
        history_rows.append({
            "Round": h["round"],
            "Accuracy (%)": h["accuracy"],
            "Loss": h["loss"],
            "Label": f"{cfg['strategy']} | {cfg['attack_type'].split('.')[-1]} | ratio={cfg['malicious_ratio']}",
        })

if history_rows:
    hist_df = pd.DataFrame(history_rows)
    fig_hist = px.line(hist_df, x="Round", y="Accuracy (%)", color="Label",
                       template="plotly_dark", title="Accuracy over Training Rounds")
    st.plotly_chart(fig_hist, use_container_width=True)

# Raw Data Table 
with st.expander("Full Results Table"):
    st.dataframe(
        filtered_df.sort_values("Accuracy (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "bft_fl_results.csv", "text/csv")

st.caption("BFT-FL Framework | University of Michigan-Dearborn | Trustworthy AI")
