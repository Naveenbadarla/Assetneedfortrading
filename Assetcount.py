import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Flex Aggregator Sizing – EVs & Batteries",
    layout="wide"
)

st.title("⚡ Flex Aggregator Sizing Tool – 15 Minute Products (DE)")

st.markdown("""
This tool helps you estimate how many **EVs** or **home batteries** you need to
reliably offer **100 kW (or more)** in German **DA / intraday markets**.

It includes:

- **Deterministic sizing**  
- **Stochastic Monte-Carlo sizing** (availability = probability)  
- **Customer segmentation** (commuter vs home-office vs fleet EVs)

---
""")


# --------------------------------------------------------------------
# Helper function: deterministic calculation
# --------------------------------------------------------------------
def required_assets(P_target, duration_h, P_asset, E_asset, availability, soc_margin):
    if P_asset <= 0 or E_asset <= 0 or availability <= 0 or soc_margin <= 0:
        return math.inf, math.inf, math.inf

    N_power = P_target / (P_asset * availability)
    N_energy = (P_target * duration_h) / (E_asset * soc_margin * availability)
    return math.ceil(max(N_power, N_energy)), N_power, N_energy


# --------------------------------------------------------------------
# Sidebar – global market settings
# --------------------------------------------------------------------
st.sidebar.header("📊 Market Settings")

P_min = st.sidebar.number_input(
    "Minimum tradable block (kW)", value=100.0, min_value=10.0, step=10.0
)

duration_h = st.sidebar.selectbox(
    "Product duration", [0.25, 0.5, 1.0], index=0,
    format_func=lambda x: f"{x} hours"
)

safety_margin = st.sidebar.slider(
    "Safety margin (%)", min_value=0, max_value=100, value=20
) / 100

P_target = P_min * (1 + safety_margin)

st.sidebar.markdown(f"**➡️ Target power = {P_target:.1f} kW**")


# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab_det, tab_stoch = st.tabs(["Deterministic Model", "Stochastic Model (Monte-Carlo)"])



# ====================================================================
# TAB 1 – Deterministic Model
# ====================================================================
with tab_det:

    st.header("🔢 Deterministic Sizing")

    asset_type = st.selectbox(
        "Choose asset type",
        ["Home battery (DE typical)", "EV (realistic V2G)", "Custom"]
    )

    if asset_type == "Home battery (DE typical)":
        P_asset = 5.5
        E_asset = 8.7
        availability = 0.9
        soc_margin = 0.7

    elif asset_type == "EV (realistic V2G)":
        P_asset = 7.0         # kW export
        E_asset = 60.0        # kWh battery
        availability = 0.30   # 30% EVs plugged in on avg
        soc_margin = 0.20     # only small SoC window allowed

    else:
        P_asset = 5.0
        E_asset = 10.0
        availability = 0.5
        soc_margin = 0.5

    col1, col2 = st.columns(2)

    with col1:
        P_asset = st.number_input("Per-asset power (kW)", value=P_asset, step=0.1)
        availability = st.slider("Availability (plugged-in probability)", 
                                 0.05, 1.0, value=availability, step=0.05)
    with col2:
        E_asset = st.number_input("Usable battery energy (kWh)", value=E_asset, step=0.5)
        soc_margin = st.slider("Usable SoC fraction", 0.05, 1.0, value=soc_margin, step=0.05)

    N_required, N_power, N_energy = required_assets(
        P_target, duration_h, P_asset, E_asset, availability, soc_margin
    )

    st.subheader("Result")

    st.metric("Minimum assets required", N_required)
    st.write(f"- Power-driven requirement: {N_power:.1f}")
    st.write(f"- Energy-driven requirement: {N_energy:.1f}")

    st.info("""
    For typical EV parameters, this model will usually be **power-constrained**, not energy-constrained.
    """)



# ====================================================================
# TAB 2 – Stochastic Model (Monte-Carlo)
# ====================================================================
with tab_stoch:

    st.header("🎲 Stochastic Sizing with Monte-Carlo Simulation")
    st.markdown("""
    In reality **not all customers behave the same**.

    - Availability is random  
    - Customers plug in at different times  
    - SoC windows vary  
    - EVs are heterogeneous  

    Here we model availability as a **probability**, and simulate thousands of
    15-minute intervals.
    """)

    st.subheader("Customer Segments")

    st.markdown("""
    Add 1–3 segments. Each segment has:
    - number of vehicles
    - per-EV power
    - battery size
    - probability of being plugged in (availability)
    - SoC window
    """)

    seg_count = st.selectbox("How many segments?", [1, 2, 3], index=2)

    segments = []

    defaults = [
        ("Commuters", 200, 0.2),
        ("Home-office", 100, 0.5),
        ("Fleet EVs", 50, 0.8)
    ]

    for i in range(seg_count):
        st.markdown(f"### Segment {i+1}")

        name, N_default, p_default = defaults[i]

        colA, colB, colC = st.columns(3)
        with colA:
            N = st.number_input(f"Vehicles in segment {i+1}", min_value=0, value=N_default)
        with colB:
            P_seg = st.number_input(f"Export power per EV (kW) – seg {i+1}", value=7.0)
        with colC:
            E_seg = st.number_input(f"Battery energy per EV (kWh) – seg {i+1}", value=60.0)

        colX, colY = st.columns(2)
        with colX:
            p = st.slider(f"Availability probability – seg {i+1}", 
                          0.0, 1.0, value=p_default, step=0.05)
        with colY:
            m = st.slider(f"SoC margin – seg {i+1}", 
                          0.05, 1.0, value=0.20, step=0.05)

        segments.append((N, P_seg, E_seg, p, m))


    st.subheader("Simulation Settings")

    sims = st.number_input("Monte-Carlo samples", min_value=100, max_value=50000, value=5000, step=500)

    run_sim = st.button("Run Monte-Carlo Simulation")

    if run_sim:
        st.write("Running simulation…")

        P_available_samples = []

        for _ in range(sims):
            P_total = 0

            for (N, P_seg, E_seg, p, m) in segments:
                # available vehicles drawn from Binomial
                A = np.random.binomial(N, p)

                # power contribution
                P_power = A * P_seg

                # energy constraint per EV
                E_total = A * E_seg * m
                P_energy = E_total / duration_h

                P_seg_tradable = min(P_power, P_energy)
                P_total += P_seg_tradable

            P_available_samples.append(P_total)

        P_available_samples = np.array(P_available_samples)

        prob_meet = (P_available_samples >= P_target).mean()

        st.subheader("Results")
        st.metric("Probability of meeting target power", f"{prob_meet*100:.1f}%")

        st.write(f"Average available power: **{P_available_samples.mean():.1f} kW**")
        st.write(f"5th percentile (very conservative): **{np.percentile(P_available_samples, 5):.1f} kW**")

        st.subheader("Distribution of Available Power")

        hist_df = pd.DataFrame({"Available power (kW)": P_available_samples})
        st.bar_chart(hist_df)

        st.info("""
        If you want **99% reliability**, increase fleet size or improve
        availability (customer incentives, workplace charging, depot fleets).
        """)

