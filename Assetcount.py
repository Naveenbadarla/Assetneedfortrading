import math
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Streamlit Page Setup
# ============================================================
st.set_page_config(
    page_title="Flex Aggregator Sizing Tool",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Flex Aggregator Sizing Tool – EVs & Home Batteries")
st.markdown("""
Estimate how many **EVs** or **home batteries** you need to reliably provide  
**15-minute tradable flexibility** (e.g., 100 kW) in DE-LU DA/ID markets.

This tool includes:
- 🔢 Deterministic sizing  
- 🎲 Monte-Carlo stochastic sizing  
- 👥 Customer segmentation  
- EV-appropriate & home-battery-appropriate segment defaults  

---
""")


# ============================================================
# Helper Functions
# ============================================================
def deterministic_required_assets(P_target, duration_h, P_asset, E_asset, availability, soc_margin):
    """Return deterministic sizing requirement."""
    if P_asset <= 0 or E_asset <= 0 or availability <= 0 or soc_margin <= 0:
        return math.inf, math.inf, math.inf

    N_power = P_target / (P_asset * availability)
    N_energy = (P_target * duration_h) / (E_asset * soc_margin * availability)

    return math.ceil(max(N_power, N_energy)), N_power, N_energy


def simulate_monte_carlo(segments, duration_h, sims, P_target):
    """Monte-Carlo simulation for total available power."""
    results = np.zeros(sims)

    for i in range(sims):
        total_power = 0
        for (N, Pseg, Eseg, p, m) in segments:
            A = np.random.binomial(int(N), float(p))

            # Power-limited
            P_power = A * float(Pseg)

            # Energy-limited
            E_total = A * float(Eseg) * float(m)
            P_energy = E_total / float(duration_h)

            total_power += min(P_power, P_energy)

        results[i] = total_power

    return {
        "samples": results,
        "mean": results.mean(),
        "p05": np.percentile(results, 5),
        "p95": np.percentile(results, 95),
        "prob_meet": (results >= P_target).mean()
    }


# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("⚙ Market Parameters")

P_min = float(st.sidebar.number_input("Minimum tradable block (kW)", value=100.0, step=10.0))
duration_h = float(st.sidebar.selectbox("Product duration", [0.25, 0.5, 1.0], index=0))
safety_margin = float(st.sidebar.slider("Safety margin (%)", 0, 100, 20)) / 100

P_target = P_min * (1 + safety_margin)
st.sidebar.success(f"Target power including margin: **{P_target:.1f} kW**")


# ============================================================
# Asset Type Selection
# ============================================================
asset_choice = st.radio(
    "Select Asset Type",
    ["EV Fleet (V2G)", "Home Battery Fleet"],
    horizontal=True
)

# Presets for asset parameters
if asset_choice == "EV Fleet (V2G)":
    default_P = 7.0
    default_E = 60.0
    default_avail = 0.30
    default_soc = 0.20
else:
    default_P = 5.5
    default_E = 8.7
    default_avail = 0.90
    default_soc = 0.70

# Force all defaults to float to avoid StreamlitMixedNumericTypesError
default_P = float(default_P)
default_E = float(default_E)
default_avail = float(default_avail)
default_soc = float(default_soc)


# ============================================================
# Tabs
# ============================================================
tab_det, tab_stoch = st.tabs(["🔢 Deterministic Model", "🎲 Stochastic Monte-Carlo Model"])


# ============================================================
# TAB 1 — Deterministic Sizing
# ============================================================
with tab_det:

    st.header("🔢 Deterministic Sizing")

    with st.expander("Asset Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            P_asset = float(st.number_input("Power per asset (kW)", value=default_P, step=0.1))
        with col2:
            E_asset = float(st.number_input("Energy per asset (kWh)", value=default_E, step=0.1))
        with col3:
            availability = float(st.slider("Availability probability", 0.05, 1.0, value=default_avail))
        with col4:
            soc_margin = float(st.slider("Usable SoC fraction", 0.05, 1.0, value=default_soc))

    N_req, N_power, N_energy = deterministic_required_assets(
        P_target, duration_h, P_asset, E_asset, availability, soc_margin
    )

    c1, c2 = st.columns(2)
    c1.metric("Required number of assets", f"{N_req}")
    c2.metric("Power-driven requirement", f"{N_power:.1f} assets")

    st.write(f"Energy-driven requirement: **{N_energy:.1f} assets**")

    st.info("""
    ✔ EV fleets are usually **power-limited**  
    ✔ Home batteries may be **energy-limited** depending on duration  
    """)


# ============================================================
# TAB 2 — Stochastic Monte-Carlo Sizing
# ============================================================
with tab_stoch:

    st.header("🎲 Monte-Carlo Stochastic Sizing")

    seg_count = st.selectbox("Number of customer segments", [1, 2, 3], index=1)

    # Different segment defaults depending on asset type
    if asset_choice == "EV Fleet (V2G)":
        predefined = [
            ("Commuters", 200.0, 0.20),
            ("Home Office Users", 100.0, 0.50),
            ("Fleet / Depot Vehicles", 50.0, 0.80),
        ]
    else:  # Home Batteries
        predefined = [
            ("Standard Households", 200.0, 0.90),
            ("PV-heavy Prosumers", 100.0, 0.95),
            ("Weekend / Low-usage Homes", 50.0, 0.70),
        ]

    segments = []

    for i in range(seg_count):

        name, N_default, p_default = predefined[i]

        st.subheader(f"Segment {i+1} — {name}")

        colA, colB, colC, colD, colE = st.columns(5)

        with colA:
            N_seg = float(st.number_input(f"{name} — number of assets",
                                          value=float(N_default), step=1.0))

        with colB:
            P_seg = float(st.number_input(f"{name} — power per asset (kW)",
                                          value=default_P))

        with colC:
            E_seg = float(st.number_input(f"{name} — energy per asset (kWh)",
                                          value=default_E))

        with colD:
            avail_seg = float(st.slider(f"{name} — availability probability",
                                        0.0, 1.0, value=float(p_default)))

        with colE:
            soc_seg = float(st.slider(f"{name} — usable SoC margin",
                                      0.05, 1.0, value=default_soc))

        segments.append((N_seg, P_seg, E_seg, avail_seg, soc_seg))

    sims = int(st.number_input("Monte-Carlo samples",
                               min_value=500, max_value=30000,
                               value=5000, step=500))

    if st.button("Run Monte-Carlo Simulation"):

        with st.spinner("Simulating thousands of intervals…"):
            results = simulate_monte_carlo(segments, duration_h, sims, P_target)

        mean = results["mean"]
        p5 = results["p05"]
        p95 = results["p95"]
        prob = results["prob_meet"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean available power", f"{mean:.1f} kW")
        c2.metric("5% worst-case (P5)", f"{p5:.1f} kW")
        c3.metric("Probability to meet target", f"{prob:.1f}%")

        st.subheader("Distribution of Available Power")
        df = pd.DataFrame({"Available Power (kW)": results["samples"]})
        st.bar_chart(df)

        st.info("""
        If probability is below **95–99%**, consider increasing assets,
        improving availability, or mixing EVs with home batteries.
        """)

