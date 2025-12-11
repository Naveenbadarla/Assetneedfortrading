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
    page_icon="⚡",
)

st.title("⚡ Flex Aggregator Sizing Tool – EVs & Home Batteries")
st.markdown(
    """
Estimate how many **EVs** or **home batteries** you need to reliably provide  
**15-minute tradable flexibility** (e.g., 100 kW) in DA/ID markets.

The app has three levels:

1. **Deterministic sizing** – quick analytic formula  
2. **Stochastic (per-interval) Monte-Carlo** – randomness & segments  
3. **Advanced full-day simulation** – time-of-day, PV/load, SoC, correlations  

---
"""
)


# ============================================================
# Global helpers
# ============================================================
DT_H = 0.25  # 15 min in hours


def deterministic_required_assets(P_target, duration_h, P_asset, E_asset, availability, soc_margin):
    """Return deterministic sizing requirement."""
    if P_asset <= 0 or E_asset <= 0 or availability <= 0 or soc_margin <= 0:
        return math.inf, math.inf, math.inf

    N_power = P_target / (P_asset * availability)
    N_energy = (P_target * duration_h) / (E_asset * soc_margin * availability)

    return math.ceil(max(N_power, N_energy)), N_power, N_energy


def simulate_monte_carlo_interval(segments, duration_h, sims, P_target):
    """Simple per-interval Monte Carlo (no time-of-day)."""
    results = np.zeros(sims)

    for i in range(sims):
        total_power = 0
        for (N, Pseg, Eseg, p, m) in segments:
            A = np.random.binomial(int(N), float(p))

            P_power = A * float(Pseg)
            E_total = A * float(Eseg) * float(m)
            P_energy = E_total / float(duration_h)

            total_power += min(P_power, P_energy)

        results[i] = total_power

    return {
        "samples": results,
        "mean": results.mean(),
        "p05": np.percentile(results, 5),
        "p95": np.percentile(results, 95),
        "prob_meet": (results >= P_target).mean(),
    }


# ============================================================
# Time-of-day curves & PV/load profiles
# ============================================================
def upsample_24_to_96(arr24):
    """Repeat each hourly value 4 times for 15-min resolution."""
    arr24 = np.asarray(arr24, dtype=float)
    return np.repeat(arr24, 4)


def ev_pattern_commuters():
    # higher at night and evening, low while away at work
    hourly = [
        0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.4, 0.2,  # 0–7
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # 8–15
        0.2, 0.4, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9,  # 16–23
    ]
    return upsample_24_to_96(hourly)


def ev_pattern_home_office():
    # fairly high all day, very high evenings
    hourly = [
        0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8,  # 0–7
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # 8–15
        0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,  # 16–23
    ]
    return upsample_24_to_96(hourly)


def ev_pattern_fleet():
    # depot fleet: very high at night, medium in the day
    hourly = [
        0.95, 0.95, 0.95, 0.95, 0.95, 0.9, 0.8, 0.7,  # 0–7
        0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,      # 8–15
        0.7, 0.8, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95, # 16–23
    ]
    return upsample_24_to_96(hourly)


def get_ev_pattern_for_segment(seg_index):
    if seg_index == 0:
        return ev_pattern_commuters()
    elif seg_index == 1:
        return ev_pattern_home_office()
    else:
        return ev_pattern_fleet()


def pv_profile_normalized():
    """
    Very simple normalized PV profile (per kWp), "summer-ish".
    Peak ~1.0 around noon.
    """
    hourly = [
        0.0, 0.0, 0.0, 0.0,   # 0–3
        0.0, 0.05, 0.15, 0.3, # 4–7
        0.5, 0.8, 1.0, 0.9,   # 8–11
        0.8, 0.6, 0.4, 0.2,   # 12–15
        0.1, 0.02, 0.0, 0.0,  # 16–19
        0.0, 0.0, 0.0, 0.0,   # 20–23
    ]
    return upsample_24_to_96(hourly)


def load_profile_household_normalized():
    """
    Rough typical household load shape normalized to 1.0 at peak.
    """
    hourly = [
        0.35, 0.3, 0.3, 0.3,    # 0–3
        0.35, 0.5, 0.7, 0.6,    # 4–7 (morning bump)
        0.4, 0.35, 0.35, 0.35,  # 8–11 (day low)
        0.4, 0.45, 0.6, 0.8,    # 12–15
        0.9, 1.0, 0.9, 0.8,     # 16–19 big evening peak
        0.7, 0.6, 0.5, 0.4,     # 20–23
    ]
    return upsample_24_to_96(hourly)


PV_PROFILE = pv_profile_normalized()  # length 96
LOAD_PROFILE = load_profile_household_normalized()  # length 96


# ============================================================
# Advanced day simulation helpers
# ============================================================
def simulate_ev_day(segments, P_target, duration_h, sims):
    """
    Full-day EV simulation with time-of-day availability.
    segments: list of (N, Pseg, Eseg, p_base, m)
    Returns stats over days: distribution of min power, fraction of shortfall intervals, etc.
    """
    num_steps = int(24 / DT_H)
    min_power_per_day = np.zeros(sims)
    shortfall_frac_per_day = np.zeros(sims)

    for s in range(sims):
        P_t = np.zeros(num_steps)

        # scenario-level "day type" factor: 0.8–1.2
        day_type_factor = np.random.normal(loc=1.0, scale=0.1)
        day_type_factor = np.clip(day_type_factor, 0.7, 1.3)

        for seg_idx, (N, Pseg, Eseg, p_base, m) in enumerate(segments):
            if N <= 0:
                continue

            pattern = get_ev_pattern_for_segment(seg_idx)  # [0,1]
            # availability prob as function of time: base * pattern * day factor
            p_t = np.clip(p_base * pattern * day_type_factor, 0.0, 1.0)

            # sample number of available EVs for each time
            A_t = np.random.binomial(int(N), p_t)

            # power and energy limits
            P_power_t = A_t * float(Pseg)
            E_total_t = A_t * float(Eseg) * float(m)
            P_energy_t = np.where(duration_h > 0, E_total_t / float(duration_h), 0.0)

            P_seg_t = np.minimum(P_power_t, P_energy_t)
            P_t += P_seg_t

        min_power_per_day[s] = P_t.min()
        shortfall_frac_per_day[s] = np.mean(P_t < P_target)

    return {
        "min_power": min_power_per_day,
        "shortfall_frac": shortfall_frac_per_day,
        "prob_full_day": np.mean(min_power_per_day >= P_target),
        "avg_shortfall_frac": shortfall_frac_per_day.mean(),
    }


def simulate_home_battery_day(segments, P_target, sims, weather_sigma):
    """
    Full-day home battery simulation with PV + load + SoC dynamics.
    segments: list of (N, Pseg, Eseg, p_avail, soc_margin, pv_kwp_per_home, daily_load_kwh)
    """
    num_steps = len(PV_PROFILE)
    min_power_per_day = np.zeros(sims)
    shortfall_frac_per_day = np.zeros(sims)

    for s in range(sims):
        P_t = np.zeros(num_steps)

        # scenario-level weather factor (shared across all homes)
        weather_factor = np.random.lognormal(mean=0.0, sigma=weather_sigma)
        weather_factor = np.clip(weather_factor, 0.3, 1.5)

        for (N, Pseg, Eseg, p_avail, m, pv_kwp, daily_kwh) in segments:
            if N <= 0:
                continue

            cap = float(Eseg)
            Pmax = float(Pseg)
            soc_min_frac = 1.0 - float(m)  # if m is "tradable margin", keep the rest
            soc_min = soc_min_frac * cap

            # Build absolute PV & load profiles for this segment (kW)
            # scale normalized profiles
            # PV peak power ~ pv_kwp
            pv_kw = PV_PROFILE * pv_kwp * weather_factor
            # scale load profile so integral over day = daily_kwh
            base_load = LOAD_PROFILE
            norm = np.sum(base_load * DT_H)
            load_kw = base_load * (daily_kwh / norm)

            # SoC trajectory for a "typical" home in the segment (no trading)
            soc = np.zeros(num_steps + 1)
            # start at 50% full as a neutral point
            soc[0] = 0.5 * cap

            for t in range(num_steps):
                net_kw = pv_kw[t] - load_kw[t]  # positive: surplus
                delta_e = net_kw * DT_H  # kWh

                if delta_e >= 0:
                    # charge if surplus
                    soc[t + 1] = min(cap, soc[t] + delta_e)
                else:
                    # discharge to cover deficit down to soc_min
                    need = -delta_e
                    available = max(soc[t] - soc_min, 0.0)
                    discharge = min(need, available)
                    soc[t + 1] = soc[t] - discharge
                    # remaining deficit is imported from grid, ignored here

            # available energy margin for trading per home at each time
            # we assume we only use a fraction m of the headroom above soc_min
            margin_energy_per_home = np.maximum(soc[1:] - soc_min, 0.0) * m
            P_energy_per_home = margin_energy_per_home / DT_H

            # randomly available homes (technical/contractual availability)
            A_t = np.random.binomial(int(N), float(p_avail), size=num_steps)

            P_power_t = A_t * Pmax
            P_energy_t = A_t * P_energy_per_home
            P_seg_t = np.minimum(P_power_t, P_energy_t)

            P_t += P_seg_t

        min_power_per_day[s] = P_t.min()
        shortfall_frac_per_day[s] = np.mean(P_t < P_target)

    return {
        "min_power": min_power_per_day,
        "shortfall_frac": shortfall_frac_per_day,
        "prob_full_day": np.mean(min_power_per_day >= P_target),
        "avg_shortfall_frac": shortfall_frac_per_day.mean(),
    }


# ============================================================
# Sidebar – market parameters
# ============================================================
st.sidebar.header("⚙ Market Parameters")

P_min = float(st.sidebar.number_input("Minimum tradable block (kW)", value=100.0, step=10.0))
duration_h = float(st.sidebar.selectbox("Product duration", [0.25, 0.5, 1.0], index=0))
safety_margin = float(st.sidebar.slider("Safety margin (%)", 0, 100, 20)) / 100

P_target = P_min * (1 + safety_margin)
st.sidebar.success(f"Target power including margin: **{P_target:.1f} kW**")


# ============================================================
# Asset Type selection
# ============================================================
asset_choice = st.radio(
    "Select Asset Type (for all tabs below)",
    ["EV Fleet (V2G)", "Home Battery Fleet"],
    horizontal=True,
)

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

default_P = float(default_P)
default_E = float(default_E)
default_avail = float(default_avail)
default_soc = float(default_soc)


# ============================================================
# Tabs
# ============================================================
tab_det, tab_stoch, tab_adv = st.tabs(
    ["🔢 Deterministic Model", "🎲 Stochastic (Single Interval)", "📅 Advanced Full-Day Simulation"]
)


# ============================================================
# TAB 1 – Deterministic
# ============================================================
with tab_det:
    st.header("🔢 Deterministic Sizing (Single Interval)")

    with st.expander("Asset Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            P_asset = float(st.number_input("Power per asset (kW)", value=default_P, step=0.1))
        with c2:
            E_asset = float(st.number_input("Energy per asset (kWh)", value=default_E, step=0.1))
        with c3:
            availability = float(
                st.slider("Availability probability (average)", 0.05, 1.0, value=default_avail)
            )
        with c4:
            soc_margin = float(
                st.slider("Usable SoC fraction for trading", 0.05, 1.0, value=default_soc)
            )

    N_req, N_power, N_energy = deterministic_required_assets(
        P_target, duration_h, P_asset, E_asset, availability, soc_margin
    )

    m1, m2 = st.columns(2)
    m1.metric("Required number of assets", f"{N_req}")
    m2.metric("Power-driven requirement", f"{N_power:.1f} assets")
    st.write(f"Energy-driven requirement: **{N_energy:.1f} assets**")

    st.info(
        "EV fleets are typically **power-limited**; "
        "home batteries can become **energy-limited** for longer durations."
    )


# ============================================================
# TAB 2 – Stochastic per-interval
# ============================================================
with tab_stoch:
    st.header("🎲 Stochastic Monte-Carlo (Single Interval)")
    st.markdown(
        "This models a **single 15-minute product** with randomness in availability, "
        "but no time-of-day structure yet."
    )

    seg_count = st.selectbox("Number of customer segments", [1, 2, 3], index=1)

    if asset_choice == "EV Fleet (V2G)":
        predefined = [
            ("Commuters", 200.0, 0.20),
            ("Home Office Users", 100.0, 0.50),
            ("Fleet / Depot Vehicles", 50.0, 0.80),
        ]
    else:
        predefined = [
            ("Standard Households", 200.0, 0.90),
            ("PV-heavy Prosumers", 100.0, 0.95),
            ("Weekend / Low-usage Homes", 50.0, 0.70),
        ]

    segments = []
    for i in range(seg_count):
        name, N_default, p_default = predefined[i]
        st.subheader(f"Segment {i+1} — {name}")
        ca, cb, cc, cd, ce = st.columns(5)
        with ca:
            N_seg = float(
                st.number_input(f"{name} — number of assets", value=float(N_default), step=1.0)
            )
        with cb:
            P_seg = float(
                st.number_input(f"{name} — power per asset (kW)", value=default_P)
            )
        with cc:
            E_seg = float(
                st.number_input(f"{name} — energy per asset (kWh)", value=default_E)
            )
        with cd:
            avail_seg = float(
                st.slider(f"{name} — availability probability", 0.0, 1.0, value=float(p_default))
            )
        with ce:
            soc_seg = float(
                st.slider(f"{name} — usable SoC margin", 0.05, 1.0, value=default_soc)
            )

        segments.append((N_seg, P_seg, E_seg, avail_seg, soc_seg))

    sims_interval = int(
        st.number_input("Monte-Carlo samples (interval)", min_value=500, max_value=30000, value=5000, step=500)
    )

    if st.button("Run Single-Interval Monte-Carlo"):
        with st.spinner("Running Monte-Carlo for one interval…"):
            res = simulate_monte_carlo_interval(segments, duration_h, sims_interval, P_target)

        mean = res["mean"]
        p5 = res["p05"]
        p95 = res["p95"]
        prob = res["prob_meet"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean available power", f"{mean:.1f} kW")
        c2.metric("5% worst-case (P5)", f"{p5:.1f} kW")
        c3.metric("Probability to meet target", f"{prob:.1f}%")

        st.subheader("Distribution of available power (single interval)")
        df = pd.DataFrame({"Available Power (kW)": res["samples"]})
        st.bar_chart(df)

        # -----------------------------
        # RESULT SUMMARY BLOCK
        # -----------------------------
        st.subheader("📘 Result Summary")
        
        prob = res["prob_meet"] * 100  # or res_day["prob_full_day"]*100 for the day simulation
        mean_power = res["mean"]       # or min_power.mean()
        p5 = res["p05"]
        
        if prob >= 97:
            color = "green"
            status = "Excellent reliability 👍"
            text = f"""
            Your fleet can **safely** deliver the {P_target:.0f} kW block in **{prob:.1f}%** of intervals.
        
            This meets typical requirements for:
            - 🔹 Day-Ahead trading  
            - 🔹 Intraday block bids  
            - 🔹 Reserve/flex markets (FCR/FFR pre-qualification levels)
        
            Your minimum power (5% worst-case = {p5:.1f} kW) is comfortably above target.
            """
        elif prob >= 90:
            color = "orange"
            status = "Reasonably reliable ⚠️"
            text = f"""
            Your fleet delivers the target in **{prob:.1f}%** of intervals.
        
            This is usually acceptable, but:
            - There is **some risk of shortfall**
            - Might not meet strict pre-qualification criteria
            - Consider adding more EVs or increasing availability
        
            5% worst-case is **{p5:.1f} kW**, which may be below the target.
            """
        elif prob >= 60:
            color = "darkorange"
            status = "Unstable performance 🟧"
            text = f"""
            Your fleet only meets the target in **{prob:.1f}%** of intervals.
        
            You have **medium-to-high shortfall risk**, meaning:
            - Not suitable for firm products
            - Likely imbalance penalties if traded
            - Need more assets or better operational availability
            """
        else:
            color = "red"
            status = "Not trade-ready ❌"
            text = f"""
            Your fleet is **not sufficient** to reliably deliver the target.
        
            - Probability to meet target: **{prob:.1f}%**
            - 5% worst-case: **{p5:.1f} kW**
            - Typically requires **much larger fleet** or **higher plug-in rate**
        
            Right now, the system would fail **most intervals**.
            """
        
        # Display colored box
        st.markdown(
            f"""
            <div style="border-left: 8px solid {color}; padding: 1em; background-color: #1e1e1e;">
                <h3 style="color:{color};">{status}</h3>
                <p style="color:white; font-size:1.1em;">{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 3 – Advanced Full-Day Simulation
# ============================================================
with tab_adv:
    st.header("📅 Advanced Full-Day Simulation")
    st.markdown(
        """
This simulates an entire **24-hour day (96 × 15-min steps)** with:

- Time-of-day behaviour (EV patterns)
- PV + household load + battery SoC (for home batteries)
- Scenario-level correlation (weather / day type)
- Probability you can **hold the target power for every interval** in the day
        """
    )

    st.subheader("Segments & behavioural parameters")

    seg_count_adv = st.selectbox("Number of segments (advanced)", [1, 2, 3], index=1)

    segments_ev = []
    segments_hb = []

    if asset_choice == "EV Fleet (V2G)":
        predefined_adv = [
            ("Commuters", 200.0, 0.20),
            ("Home Office Users", 100.0, 0.50),
            ("Fleet / Depot Vehicles", 50.0, 0.80),
        ]
    else:
        predefined_adv = [
            ("Standard Households", 200.0, 0.90),
            ("PV-heavy Prosumers", 100.0, 0.95),
            ("Weekend / Low-usage Homes", 50.0, 0.70),
        ]

    for i in range(seg_count_adv):
        name, N_default, p_default = predefined_adv[i]
        st.subheader(f"Segment {i+1} — {name}")

        if asset_choice == "EV Fleet (V2G)":
            ca, cb, cc, cd, ce = st.columns(5)
            with ca:
                N_seg = float(
                    st.number_input(f"{name} — number of EVs", value=float(N_default), step=1.0, key=f"adv_N_ev_{i}")
                )
            with cb:
                P_seg = float(
                    st.number_input(f"{name} — power per EV (kW)", value=default_P, key=f"adv_P_ev_{i}")
                )
            with cc:
                E_seg = float(
                    st.number_input(f"{name} — battery per EV (kWh)", value=default_E, key=f"adv_E_ev_{i}")
                )
            with cd:
                avail_seg = float(
                    st.slider(
                        f"{name} — average availability",
                        0.0,
                        1.0,
                        value=float(p_default),
                        key=f"adv_avail_ev_{i}",
                    )
                )
            with ce:
                soc_seg = float(
                    st.slider(
                        f"{name} — tradable SoC margin",
                        0.05,
                        1.0,
                        value=default_soc,
                        key=f"adv_soc_ev_{i}",
                    )
                )

            segments_ev.append((N_seg, P_seg, E_seg, avail_seg, soc_seg))

        else:  # Home batteries
            ca, cb, cc, cd, ce, cf, cg = st.columns(7)
            with ca:
                N_seg = float(
                    st.number_input(
                        f"{name} — number of homes", value=float(N_default), step=1.0, key=f"adv_N_hb_{i}"
                    )
                )
            with cb:
                P_seg = float(
                    st.number_input(
                        f"{name} — inverter power (kW)",
                        value=default_P,
                        key=f"adv_P_hb_{i}",
                    )
                )
            with cc:
                E_seg = float(
                    st.number_input(
                        f"{name} — battery capacity (kWh)",
                        value=default_E,
                        key=f"adv_E_hb_{i}",
                    )
                )
            with cd:
                avail_seg = float(
                    st.slider(
                        f"{name} — technical availability",
                        0.0,
                        1.0,
                        value=float(p_default),
                        key=f"adv_avail_hb_{i}",
                    )
                )
            with ce:
                soc_seg = float(
                    st.slider(
                        f"{name} — tradable SoC margin",
                        0.05,
                        1.0,
                        value=default_soc,
                        key=f"adv_soc_hb_{i}",
                    )
                )
            with cf:
                pv_kwp = float(
                    st.number_input(
                        f"{name} — PV size per home (kWp)",
                        value=7.0,
                        step=0.5,
                        key=f"adv_pv_hb_{i}",
                    )
                )
            with cg:
                daily_load = float(
                    st.number_input(
                        f"{name} — daily load per home (kWh)",
                        value=12.0,
                        step=0.5,
                        key=f"adv_load_hb_{i}",
                    )
                )

            segments_hb.append((N_seg, P_seg, E_seg, avail_seg, soc_seg, pv_kwp, daily_load))

    sims_day = int(
        st.number_input(
            "Monte-Carlo samples (days)",
            min_value=200,
            max_value=5000,
            value=1000,
            step=200,
            help="Each sample is a full 24-hour day with its own random behaviour.",
        )
    )

    if asset_choice == "Home Battery Fleet":
        weather_sigma = float(
            st.slider(
                "Weather variability (PV, lognormal σ)",
                0.05,
                0.8,
                value=0.3,
                help="Higher = more volatile PV from day to day.",
            )
        )
    else:
        weather_sigma = None  # not used

    if st.button("Run Full-Day Simulation"):
        if asset_choice == "EV Fleet (V2G)":
            with st.spinner("Simulating full EV days with time-of-day patterns…"):
                res_day = simulate_ev_day(segments_ev, P_target, duration_h, sims_day)
        else:
            with st.spinner("Simulating full home-battery days with PV, load & SoC…"):
                res_day = simulate_home_battery_day(segments_hb, P_target, sims_day, weather_sigma)

        prob_full = res_day["prob_full_day"] * 100
        avg_shortfall_frac = res_day["avg_shortfall_frac"] * 100
        min_power = res_day["min_power"]

        c1, c2 = st.columns(2)
        c1.metric("Probability to hold target ALL DAY", f"{prob_full:.1f}%")
        c2.metric("Average fraction of intervals with shortfall", f"{avg_shortfall_frac:.1f}%")

        st.subheader("Distribution of minimum available power over the day")
        df_min = pd.DataFrame({"Minimum daily power (kW)": min_power})
        st.bar_chart(df_min)

        st.info(
            "If you want **very reliable products (e.g. 99% probability)**, "
            "you can increase fleet size, improve availability, "
            "or combine EVs with home batteries in separate runs."
        )
