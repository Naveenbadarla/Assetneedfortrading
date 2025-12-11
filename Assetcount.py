import math
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Flex Aggregator Sizing Tool",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Flex Aggregator Sizing Tool – EVs & Home Batteries")
st.markdown("""
This tool helps you estimate how many **EVs** or **home batteries** you need to
offer **15-minute tradable flexibility** (e.g., 100 kW) in the DA/ID German market.

Use deterministic or stochastic sizing, and model customer heterogeneity.
---
""")


# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def deterministic_required_assets(P_target, duration_h, P_asset, E_asset, availability, soc_margin):
    """Return deterministic sizing requirement."""
    if P_asset <= 0 or E_asset <= 0 or availability <= 0 or soc_margin <= 0:
        return math.inf, math.inf, math.inf
    N_power = P_target / (P_asset * availability)
    N_energy = (P_target * duration_h) / (E_asset * soc_margin * availability)
    return math.ceil(max(N_power, N_energy)), N_power, N_energy


def simulate_monte_carlo(segments, duration_h, sims, P_target):
    """Monte-Carlo simulation for total available power."""
    results = []

    for _ in range(sims):
        total_power = 0
        for (N, Pseg, Eseg, p, m) in segments:
            A = np.random.binomial(N, p)
            power_limit = A * Pseg
            energy_limit = (A * Eseg * m) / duration_h
            total_power += min(power_limit, energy_limit)
        results.append(total_power)

    arr = np.array(results)
    return {
        "mean": arr.mean(),
        "p05": np.percentile(arr, 5),
        "p95": np.percentile(arr, 95),
        "prob_meet": (arr >= P_target).mean(),
        "samples": arr
    }


# --------------------------------------------------------------------
# Sidebar Settings
# --------------------------------------------------------------------
st.sidebar.header("⚙ Market Parameters")

P_min = st.sidebar.number_input("Minimum tradable block (kW)", value=100.0, step=10.0)
duration_h = st.sidebar.selectbox("Product duration", [0.25, 0.5, 1.0], index=0)
safety_margin = st.sidebar.slider("Safety margin (%)", 0, 100, 20) / 100
P_target = P_min * (1 + safety_margin)

st.sidebar.success(f"Target: **{P_target:.1f} kW**")


# --------------------------------------------------------------------
# Choose Asset Type
# --------------------------------------------------------------------
asset_choice = st.radio(
    "Select Asset Type",
    ["EV Fleet (V2G)", "Home Battery Fleet"],
    horizontal=True
)

if asset_choice == "EV Fleet (V2G)":
    default_P = 7.0
    default_E = 60
    default_avail = 0.30
    default_soc = 0.20
else:
    default_P = 5.5
    default_E = 8.7
    default_avail = 0.90
    default_soc = 0.70


# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab_det, tab_stoch = st.tabs(["🔢 Deterministic Model", "🎲 Stochastic Monte-Carlo Model"])


# ====================================================================
# TAB 1 – Deterministic Model
# ====================================================================
with tab_det:

    st.header("Deterministic Sizing")

    with st.expander("Asset Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            P_asset = st.number_input("Power per asset (kW)", value=default_P, step=0.1)
        with col2:
            E_asset = st.number_input("Energy per asset (kWh)", value=default_E, step=1.0)
        with col3:
            availability = st.slider("Availability probability", 0.05, 1.0, value=default_avail)
        with col4:
            soc_margin = st.slider("SoC Margin % Usable", 0.05, 1.0, value=default_soc)

    N_req, N_power, N_energy = deterministic_required_assets(
        P_target, duration_h, P_asset, E_asset, availability, soc_margin
    )

    # Metrics Row
    c1, c2 = st.columns(2)
    c1.metric("Required Assets", N_req)
    c2.metric("Power-driven requirement", f"{N_power:.1f}")

    st.write(f"Energy-driven requirement: **{N_energy:.1f}** assets")

    st.info("For EVs, sizing is usually power-limited. For home batteries, both constraints matter.")


# ====================================================================
# TAB 2 – Stochastic Monte-Carlo Model
# ====================================================================
with tab_stoch:

    st.header("Monte-Carlo Stochastic Sizing")
    st.markdown("""
    You can use **segments** to represent different customer types, each with its own availability.
    """)

    seg_count = st.selectbox("Number of customer segments", [1, 2, 3], index=1)

    segments = []
    defaults = [
        ("Commuters", 200, 0.20),
        ("Home-office", 100, 0.50),
        ("Fleet EVs / Always-plugged", 50, 0.80)
    ]

    for i in range(seg_count):
        name, N_default, p_default = defaults[i]

        st.subheader(f"Segment {i+1} – {name}")

        colA, colB, colC, colD, colE = st.columns(5)

        with colA:
            N_seg = st.number_input(f"{name} – number of assets", min_value=0, value=N_default)
        with colB:
            P_seg = st.number_input(f"{name} – power per asset (kW)", value=default_P)
        with colC:
            E_seg = st.number_input(f"{name} – energy per asset (kWh)", value=default_E)
        with colD:
            avail_seg = st.slider(f"{name} – availability probability", 0.0, 1.0, value=p_default)
        with colE:
            soc_seg = st.slider(f"{name} – usable SoC margin", 0.05, 1.0, value=default_soc)

        segments.append((N_seg, P_seg, E_seg, avail_seg, soc_seg))

    sims = st.number_input("Monte-Carlo samples", min_value=500, max_value=30000, value=5000, step=500)

    if st.button("Run Simulation"):
        with st.spinner("Running Monte-Carlo…"):
            results = simulate_monte_carlo(segments, duration_h, sims, P_target)

        mean = results["mean"]
        p5 = results["p05"]
        p95 = results["p95"]
        prob = results["prob_meet"] * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean available power", f"{mean:.1f} kW")
        m2.metric("5th percentile", f"{p5:.1f} kW")
        m3.metric("Probability of meeting target", f"{prob:.1f}%")

        st.subheader("Distribution of available power")
        df = pd.DataFrame({"Available Power (kW)": results["samples"]})
        st.bar_chart(df)

        st.info("""
        If probability is < 95–99%, increase fleet size or availability,
        or add home batteries to stabilize the portfolio.
        """)

