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
2. **Stochastic (single-interval) Monte-Carlo** – randomness & segments  
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
    hourly = [
        0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.4, 0.2,  # 0–7
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # 8–15
        0.2, 0.4, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9,  # 16–23
    ]
    return upsample_24_to_96(hourly)


def ev_pattern_home_office():
    hourly = [
        0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8,  # 0–7
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # 8–15
        0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,  # 16–23
    ]
    return upsample_24_to_96(hourly)


def ev_pattern_fleet():
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
    """Very simple normalized PV profile (per kWp)."""
    hourly = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.05, 0.15, 0.3,
        0.5, 0.8, 1.0, 0.9,
        0.8, 0.6, 0.4, 0.2,
        0.1, 0.02, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]
    return upsample_24_to_96(hourly)


def load_profile_household_normalized():
    """Rough typical household load shape normalized to 1.0 at peak."""
    hourly = [
        0.35, 0.3, 0.3, 0.3,
        0.35, 0.5, 0.7, 0.6,
        0.4, 0.35, 0.35, 0.35,
        0.4, 0.45, 0.6, 0.8,
        0.9, 1.0, 0.9, 0.8,
        0.7, 0.6, 0.5, 0.4,
    ]
    return upsample_24_to_96(hourly)


PV_PROFILE = pv_profile_normalized()
LOAD_PROFILE = load_profile_household_normalized()


# ============================================================
# Advanced day simulation helpers
# ============================================================
def simulate_ev_day(segments, P_target, duration_h, sims):
    """
    Full-day EV simulation with time-of-day availability.
    segments: list of (N, Pseg, Eseg, p_base, m)
    """
    num_steps = int(24 / DT_H)
    min_power_per_day = np.zeros(sims)
    shortfall_frac_per_day = np.zeros(sims)
    success_intervals_per_day = np.zeros(sims, dtype=int)

    for s in range(sims):
        P_t = np.zeros(num_steps)

        # scenario-level factor for "busy vs quiet day"
        day_type_factor = np.random.normal(loc=1.0, scale=0.1)
        day_type_factor = np.clip(day_type_factor, 0.7, 1.3)

        for seg_idx, (N, Pseg, Eseg, p_base, m) in enumerate(segments):
            if N <= 0:
                continue

            pattern = get_ev_pattern_for_segment(seg_idx)
            p_t = np.clip(p_base * pattern * day_type_factor, 0.0, 1.0)

            A_t = np.random.binomial(int(N), p_t)

            P_power_t = A_t * float(Pseg)
            E_total_t = A_t * float(Eseg) * float(m)
            P_energy_t = np.where(duration_h > 0, E_total_t / float(duration_h), 0.0)

            P_seg_t = np.minimum(P_power_t, P_energy_t)
            P_t += P_seg_t

        min_power_per_day[s] = P_t.min()
        shortfall_frac_per_day[s] = np.mean(P_t < P_target)
        success_intervals_per_day[s] = np.sum(P_t >= P_target)

    return {
        "min_power": min_power_per_day,
        "shortfall_frac": shortfall_frac_per_day,
        "prob_full_day": np.mean(min_power_per_day >= P_target),
        "avg_shortfall_frac": shortfall_frac_per_day.mean(),
        "success_intervals": success_intervals_per_day,
    }


def simulate_home_battery_day(segments, P_target, sims, weather_sigma):
    """
    Full-day home battery simulation with PV + load + SoC dynamics.
    segments: list of
      (N, Pseg, Eseg, p_avail, soc_margin, pv_kwp_per_home, daily_load_kwh)
    """
    num_steps = len(PV_PROFILE)
    min_power_per_day = np.zeros(sims)
    shortfall_frac_per_day = np.zeros(sims)
    success_intervals_per_day = np.zeros(sims, dtype=int)

    for s in range(sims):
        P_t = np.zeros(num_steps)

        # scenario-level PV factor
        weather_factor = np.random.lognormal(mean=0.0, sigma=weather_sigma)
        weather_factor = np.clip(weather_factor, 0.3, 1.5)

        for (N, Pseg, Eseg, p_avail, m, pv_kwp, daily_kwh) in segments:
            if N <= 0:
                continue

            cap = float(Eseg)
            Pmax = float(Pseg)
            soc_min_frac = 1.0 - float(m)
            soc_min = soc_min_frac * cap

            pv_kw = PV_PROFILE * pv_kwp * weather_factor
            base_load = LOAD_PROFILE
            norm = np.sum(base_load * DT_H)
            load_kw = base_load * (daily_kwh / norm)

            soc = np.zeros(num_steps + 1)
            soc[0] = 0.5 * cap  # start at 50%

            for t in range(num_steps):
                net_kw = pv_kw[t] - load_kw[t]
                delta_e = net_kw * DT_H

                if delta_e >= 0:
                    soc[t + 1] = min(cap, soc[t] + delta_e)
                else:
                    need = -delta_e
                    available = max(soc[t] - soc_min, 0.0)
                    discharge = min(need, available)
                    soc[t + 1] = soc[t] - discharge

            margin_energy_per_home = np.maximum(soc[1:] - soc_min, 0.0) * m
            P_energy_per_home = margin_energy_per_home / DT_H

            A_t = np.random.binomial(int(N), float(p_avail), size=num_steps)

            P_power_t = A_t * Pmax
            P_energy_t = A_t * P_energy_per_home
            P_seg_t = np.minimum(P_power_t, P_energy_t)

            P_t += P_seg_t

        min_power_per_day[s] = P_t.min()
        shortfall_frac_per_day[s] = np.mean(P_t < P_target)
        success_intervals_per_day[s] = np.sum(P_t >= P_target)

    return {
        "min_power": min_power_per_day,
        "shortfall_frac": shortfall_frac_per_day,
        "prob_full_day": np.mean(min_power_per_day >= P_target),
        "avg_shortfall_frac": shortfall_frac_per_day.mean(),
        "success_intervals": success_intervals_per_day,
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
        st.number_input(
            "Monte-Carlo samples (interval)", min_value=500, max_value=30000, value=5000, step=500
        )
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

        # Simple summary for single-interval case
        st.subheader("📘 Result Summary (Single Interval)")
        if prob >= 97:
            summary = (
                f"Your fleet can safely deliver the {P_target:.0f} kW block "
                f"in **{prob:.1f}%** of intervals. This is excellent reliability."
            )
        elif prob >= 90:
            summary = (
                f"Your fleet delivers the target in **{prob:.1f}%** of intervals. "
                "Usually okay, but some shortfall risk remains."
            )
        elif prob >= 60:
            summary = (
                f"Your fleet only meets the target in **{prob:.1f}%** of intervals. "
                "This is unstable and not suitable for firm products."
            )
        else:
            summary = (
                f"Reliability is only **{prob:.1f}%**. "
                "The fleet is far too small or availability too low for firm trading."
            )
        st.write(summary)


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
                    st.number_input(
                        f"{name} — number of EVs",
                        value=float(N_default),
                        step=1.0,
                        key=f"adv_N_ev_{i}",
                    )
                )
            with cb:
                P_seg = float(
                    st.number_input(
                        f"{name} — power per EV (kW)",
                        value=default_P,
                        key=f"adv_P_ev_{i}",
                    )
                )
            with cc:
                E_seg = float(
                    st.number_input(
                        f"{name} — battery per EV (kWh)",
                        value=default_E,
                        key=f"adv_E_ev_{i}",
                    )
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
                        f"{name} — number of homes",
                        value=float(N_default),
                        step=1.0,
                        key=f"adv_N_hb_{i}",
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
        success_counts = res_day["success_intervals"]

        # tradable intervals stats
        avg_success = success_counts.mean()
        median_success = np.median(success_counts)
        p10_success = np.percentile(success_counts, 10)
        p90_success = np.percentile(success_counts, 90)

        avg_hours = avg_success * DT_H
        median_hours = median_success * DT_H
        p10_hours = p10_success * DT_H
        p90_hours = p90_success * DT_H

        c1, c2 = st.columns(2)
        c1.metric("Probability to hold target ALL DAY", f"{prob_full:.1f}%")
        c2.metric("Average fraction of intervals with shortfall", f"{avg_shortfall_frac:.1f}%")

        st.subheader("Distribution of minimum available power over the day")
        df_min = pd.DataFrame({"Minimum daily power (kW)": min_power})
        st.bar_chart(df_min)

        st.subheader("📊 Tradable intervals per day (at target power)")
        c3, c4, c5 = st.columns(3)
        c3.metric(
            "Average successful intervals",
            f"{avg_success:.1f} of 96",
            help="Intervals where available power ≥ target.",
        )
        c4.metric(
            "Median successful intervals",
            f"{median_success:.0f} of 96",
        )
        c5.metric(
            "Range (10–90% of days)",
            f"{p10_success:.0f} – {p90_success:.0f} intervals",
        )
        st.write(
            f"On a **typical day**, you can trade at least the target power in "
            f"about **{avg_success:.1f} intervals**, which is roughly "
            f"**{avg_hours:.1f} hours per day**.\n\n"
            f"In half of all days (median), you get about **{median_success:.0f} intervals** "
            f"({median_hours:.1f} hours)."
        )

        st.subheader("📘 Detailed Result Summary (Full Day)")
        summary_lines = []

        if prob_full < 5:
            summary_lines.append(
                f"- The fleet almost **never** supports a firm 24-hour product "
                f"(only **{prob_full:.1f}%** of days can hold the target all day)."
            )
        elif prob_full < 50:
            summary_lines.append(
                f"- The fleet sometimes supports a full-day product "
                f"(**{prob_full:.1f}%** of days), but reliability is too low for firm trading."
            )
        else:
            summary_lines.append(
                f"- The fleet can hold the target all day on **{prob_full:.1f}%** of days. "
                "This may be acceptable for some products."
            )

        summary_lines.append(
            f"- On average you have **{avg_shortfall_frac:.1f}%** of intervals with a shortfall. "
            "Those intervals would cause imbalance if you committed to firm delivery."
        )

        summary_lines.append(
            f"- You can typically place trades in about **{avg_hours:.1f} hours per day** "
            f"at the full target power. On bad days (10th percentile) this drops to "
            f"**{p10_hours:.1f} hours**, and on good days (90th percentile) it rises "
            f"to **{p90_hours:.1f} hours**."
        )

        if avg_success < 40:
            summary_lines.append(
                "- This looks more like a **daytime or peak-hour** flexibility portfolio, "
                "not a 24/7 firm capacity product."
            )
        else:
            summary_lines.append(
                "- The number of successful intervals is quite high; with some tuning, "
                "this fleet might support longer or more continuous products."
            )

        st.write("\n".join(summary_lines))

        st.info(
            "If you want very reliable products (e.g. 99% probability), you can increase fleet size, "
            "improve availability, increase battery size, or combine EVs with home batteries."
        )
