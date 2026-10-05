"""
Blast Viewer — Web (V1)
Air overpressure grid map from one or more blasts, per AS2187.2.
 
Direct port of the map_pressure() logic from the original
Blast_viewer.py (tkinter/matplotlib desktop app), extended to support
multiple charges with delay timing.
 
Multiple-charge handling:
  - Charges whose delay times fall within a "simultaneity window" (default
    8 ms, per AS2187.2's treatment of near-instantaneous detonations) are
    treated as firing together: their overpressure contributions (in kPa,
    linear amplitude) are SUMMED at each grid point — a conservative,
    in-phase worst-case combination.
  - Charges on genuinely different delays are independent events: the map
    shows the worst-case ENVELOPE (max) across delay groups, since they
    don't occur at the same instant.
 
Color scale is FIXED (not auto-ranged per blast) so different charge
weights / distances are visually comparable against the same real-world
reference. Regulatory thresholds from AS2187.2 / ANZEC guidelines are
overlaid as contour lines:
  115 dB(L) — may be exceeded on up to 5% of blasts in a 12-month period
  120 dB(L) — must not be exceeded for routine blasting
  133 dB(L) — absolute limit (infrequent blasting / structural)
"""
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
st.set_page_config(page_title="Blast Viewer — Overpressure Map", layout="wide")
 
st.title("Blast Viewer — Air Overpressure Map (V1)")
st.caption("AS 2187.2 grid mapping — air overpressure, delay-aware combination")
 
# Regulatory reference levels (dB(L)), per ANZEC / AS2187.2 guidance
LIMITS_DBL = {
    "115 dB(L) — 5% exceedance limit": 115.0,
    "120 dB(L) — routine blasting limit": 120.0,
    "133 dB(L) — absolute limit": 133.0,
}
 
def dbl_to_kpa(dbl):
    # inverse of: dBL = 20*log10(1000*kPa/0.00002)
    return 2e-8 * (10 ** (dbl / 20))
 
# ---------------------------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Charges")
    st.caption("Add or remove rows for extra charges (blast holes), each with "
               "its own delay time.")
    default_charges = pd.DataFrame({
        "X (m)": [0.0],
        "Y (m)": [0.0],
        "Charge (kg)": [5.0],
        "Delay (ms)": [0],
    })
    charges_df = st.data_editor(
        default_charges, num_rows="dynamic", use_container_width=True,
        key="charges_editor",
    )
    charges_df = charges_df.dropna()
    charges_df = charges_df[charges_df["Charge (kg)"] > 0]
 
    simultaneity_window = st.number_input(
        "Simultaneity window (ms)", min_value=0, value=8, step=1,
        help="Charges whose delay times fall within this window of each other "
             "are treated as detonating together and their overpressure is "
             "summed. AS2187.2 commonly uses 8 ms as the threshold for "
             "'instantaneous'. Charges outside this window are independent "
             "events, combined as a worst-case envelope (max)."
    )
 
    st.header("Site constants (detonation properties)")
    st.caption("From your AS2187.2 regression / site calibration, or USBM RI8505 defaults. "
               "Shared across all charges (same site).")
    k_site = st.number_input("k (site constant)", value=516.0, step=1.0)
    a_site = st.number_input("a (site exponent)", value=-1.45, step=0.01, format="%.3f")
 
    st.header("Map settings")
    distance = st.number_input("Map half-size (m)", min_value=10, value=200, step=10)
    grid_size = st.number_input("Grid resolution (m)", min_value=1, value=20, step=1)
    color_levels = st.slider("Color divisions", min_value=5, max_value=60, value=25)
    unit = st.radio("Units", ["dBL", "kPa"], horizontal=True)
 
    st.header("Color scale (fixed, not auto-ranged)")
    st.caption("Kept constant so different blasts are visually comparable "
               "against the same real-world reference.")
    if unit == "dBL":
        scale_min = st.number_input("Scale min (dBL)", value=60.0, step=5.0)
        scale_max = st.number_input("Scale max (dBL)", value=180.0, step=5.0)
    else:
        scale_min = st.number_input("Scale min (kPa)", value=0.0001, step=0.0001, format="%.4f")
        scale_max = st.number_input("Scale max (kPa)", value=10.0, step=1.0)
 
    show_limits = st.checkbox("Show AS2187.2 compliance limits", value=True)
 
if charges_df.empty:
    st.warning("Add at least one charge in the sidebar table to see a map.")
    st.stop()
 
# ---------------------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------------------
def overpressure_kpa(dist, charge, k, a):
    return ((dist / (charge ** (1 / 3))) ** a) * k
 
 
def kpa_to_db(kpa):
    return 20 * np.log10((1000 * kpa) / 0.00002)
 
 
def group_by_delay(df, window):
    """Sequentially cluster charges whose delay times fall within `window`
    ms of their neighbour, sorted ascending. Returns list of index-groups."""
    df_sorted = df.sort_values("Delay (ms)").reset_index(drop=True)
    groups = []
    current = [0]
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, "Delay (ms)"] - df_sorted.loc[current[-1], "Delay (ms)"] <= window:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return df_sorted, groups
 
 
N = distance
steps = int((2 * N) // grid_size)
x = np.linspace(-N, N, steps + 1)
y = np.linspace(-N, N, steps + 1)
X, Y = np.meshgrid(x, y)
 
df_sorted, delay_groups = group_by_delay(charges_df, simultaneity_window)
 
# Within each delay group: sum kPa contributions (simultaneous, in-phase).
# Across delay groups: worst-case envelope (max), since they're independent events.
group_kpa_maps = []
for group_idx in delay_groups:
    kpa_sum = np.zeros_like(X)
    for i in group_idx:
        row = df_sorted.loc[i]
        R = np.sqrt((row["X (m)"] - X) ** 2 + (row["Y (m)"] - Y) ** 2)
        R = np.where(R == 0, 1e-6, R)
        kpa_sum += overpressure_kpa(R, row["Charge (kg)"], k_site, a_site)
    group_kpa_maps.append(kpa_sum)
 
Z_kpa = np.max(np.stack(group_kpa_maps, axis=0), axis=0)
Z = Z_kpa if unit == "kPa" else kpa_to_db(Z_kpa)
log_color = (unit == "kPa")
 
# ---------------------------------------------------------------------------
# PLOT — FIXED color range (from sidebar), clipped so out-of-range values
# still render at the boundary color instead of breaking the scale.
# ---------------------------------------------------------------------------
if log_color:
    zmin = np.log10(max(scale_min, 1e-9))
    zmax = np.log10(max(scale_max, scale_min * 10))
    Zplot = np.log10(np.clip(Z, 10 ** zmin, 10 ** zmax))
else:
    zmin, zmax = scale_min, scale_max
    Zplot = np.clip(Z, zmin, zmax)
 
zsize = (zmax - zmin) / color_levels if zmax > zmin else 1.0
 
if log_color:
    lo, hi = int(np.floor(zmin)), int(np.ceil(zmax))
    tickvals = list(range(lo, hi + 1))
    ticktext = [f"{10 ** v:g}" for v in tickvals]
    colorbar = dict(title="kPa", tickvals=tickvals, ticktext=ticktext)
else:
    colorbar = dict(title="dBL")
 
fig = go.Figure(
    data=go.Contour(
        x=x, y=y, z=Zplot,
        colorscale="Jet",
        contours=dict(start=zmin, end=zmax, size=zsize, coloring="heatmap"),
        colorbar=colorbar,
        hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>value=%{z:.3f}<extra></extra>",
    )
)
 
# Regulatory compliance contour lines
if show_limits:
    line_styles = ["dot", "dash", "solid"]
    for (label, dbl_val), style in zip(LIMITS_DBL.items(), line_styles):
        level = dbl_val if unit == "dBL" else dbl_to_kpa(dbl_val)
        plot_level = np.log10(level) if log_color else level
        if Z.min() <= level <= Z.max():
            fig.add_trace(go.Contour(
                x=x, y=y, z=Zplot,
                contours=dict(start=plot_level, end=plot_level, size=0.0001, coloring="lines"),
                line=dict(color="black", width=2, dash=style),
                showscale=False,
                name=label,
                hoverinfo="skip",
            ))
 
# Blast location markers, one per charge, labelled with delay
for i, row in df_sorted.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["X (m)"]], y=[row["Y (m)"]], mode="markers+text",
        marker=dict(color="white", size=10, line=dict(color="black", width=1)),
        text=[f"C{i+1}"], textposition="top center",
        name=f"C{i+1}: {row['Charge (kg)']:g} kg @ {row['Delay (ms)']:g} ms",
        showlegend=False,
    ))
 
fig.update_layout(
    title="Overpressure Map (fixed scale, delay-aware combination)",
    xaxis_title="Distance (m)",
    yaxis_title="Distance (m)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    width=800, height=700,
)
 
st.plotly_chart(fig, use_container_width=True)
 
if len(charges_df) > 1:
    group_summary = ", ".join(
        f"group {gi+1}: {[f'C{i+1}' for i in g]}" for gi, g in enumerate(delay_groups)
    )
    st.caption(
        f"{len(charges_df)} charges across {len(delay_groups)} delay group(s) "
        f"({group_summary}). Charges within {simultaneity_window} ms of each other "
        "are summed (simultaneous); different groups are combined as a worst-case "
        "envelope (max), since they fire at different times."
    )
 
if show_limits:
    st.caption(
        "Dotted / dashed / solid black lines mark the AS2187.2 / ANZEC compliance "
        "thresholds: 115 dB(L) (5% exceedance), 120 dB(L) (routine blasting limit), "
        "133 dB(L) (absolute limit). A line only appears if that threshold falls "
        "within the mapped area for this blast."
    )
 
with st.expander("Formula reference"):
    st.latex(r"kPa = \left(\frac{R}{Q^{1/3}}\right)^{a} \cdot k")
    st.latex(r"dBL = 20 \log_{10}\left(\frac{1000 \cdot kPa}{0.00002}\right)")
    st.caption(
        "R = distance from blast (m), Q = charge weight per delay (kg), "
        "k/a = site constants. Model per AS2187.2, compiled from USBM RI8505. "
        "Charges within the simultaneity window: kPa contributions summed "
        "(linear, in-phase worst case). Different delay groups: worst-case "
        "envelope (max) across groups."
    )