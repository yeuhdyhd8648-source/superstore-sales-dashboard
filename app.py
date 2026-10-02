"""
Superstore Sales Intelligence - single-file Streamlit dashboard
Run locally :  streamlit run app.py
Data        :  cleaned_supermarket_data.csv next to this file (header row optional)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Superstore Sales Intelligence", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ───────────────────────────── constants & theme ─────────────────────────────
COLS = ["Row_ID", "Order_ID", "Order_Date", "Ship_Date", "Ship_Mode", "Customer_ID",
        "Customer_Name", "Segment", "Country", "City", "State", "Postal_Code", "Region",
        "Product_ID", "Category", "Sub_Category", "Product_Name", "Sales"]
DATA_FILES = ["cleaned_supermarket_data.csv", "data/cleaned_supermarket_data.csv", "supermarket.csv"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
INDIGO, TEAL, AMBER, ROSE, GREEN, SLATE = "#4F46E5", "#14B8A6", "#F59E0B", "#F43F5E", "#10B981", "#64748B"
CAT_COLORS = {"Technology": INDIGO, "Furniture": TEAL, "Office Supplies": AMBER}
BAND_ORDER = ["Low (<$50)", "Medium ($50-500)", "High (>=$500)"]
STATE_ABBR = dict(x.split(":") for x in (
    "Alabama:AL,Alaska:AK,Arizona:AZ,Arkansas:AR,California:CA,Colorado:CO,Connecticut:CT,Delaware:DE,"
    "District of Columbia:DC,Florida:FL,Georgia:GA,Hawaii:HI,Idaho:ID,Illinois:IL,Indiana:IN,Iowa:IA,"
    "Kansas:KS,Kentucky:KY,Louisiana:LA,Maine:ME,Maryland:MD,Massachusetts:MA,Michigan:MI,Minnesota:MN,"
    "Mississippi:MS,Missouri:MO,Montana:MT,Nebraska:NE,Nevada:NV,New Hampshire:NH,New Jersey:NJ,"
    "New Mexico:NM,New York:NY,North Carolina:NC,North Dakota:ND,Ohio:OH,Oklahoma:OK,Oregon:OR,"
    "Pennsylvania:PA,Rhode Island:RI,South Carolina:SC,South Dakota:SD,Tennessee:TN,Texas:TX,Utah:UT,"
    "Vermont:VT,Virginia:VA,Washington:WA,West Virginia:WV,Wisconsin:WI,Wyoming:WY").split(","))

st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1400px}
.hero{padding:1.5rem 2rem;border-radius:18px;margin-bottom:1.1rem;color:#fff;
      background:linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#14B8A6 130%)}
.hero .t{font-size:2rem;font-weight:800;letter-spacing:-.02em}
.hero .s{opacity:.9;margin-top:.2rem}
.kpi{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:.95rem 1.1rem;
     background:rgba(128,128,128,.06);height:100%}
.kpi .l{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;opacity:.65}
.kpi .v{font-size:1.8rem;font-weight:750;line-height:1.25}
.kpi .d{font-size:.8rem;font-weight:600}
.up{color:#10B981}.down{color:#F43F5E}.flat{opacity:.6;font-weight:400}
.story{border-radius:14px;padding:1.2rem 1.4rem;margin:.3rem 0 1.1rem;font-size:1.05rem;line-height:1.65;
       background:rgba(79,70,229,.09);border:1px solid rgba(79,70,229,.3)}
.ins{border-left:4px solid;border-radius:10px;padding:.75rem 1rem;margin-bottom:.65rem;
     background:rgba(128,128,128,.06);font-size:.93rem;line-height:1.5}
.ins b.h{display:block;margin-bottom:.1rem}
.good{border-color:#10B981}.warn{border-color:#F59E0B}.act{border-color:#4F46E5}
div[data-testid="stTabs"] button{font-weight:600}
</style>""", unsafe_allow_html=True)


# ───────────────────────────── helpers ─────────────────────────────
def html(s: str) -> None:
    """Render HTML; '$' is escaped so Streamlit never mistakes prices for LaTeX."""
    st.markdown(s.replace("$", "&#36;"), unsafe_allow_html=True)


def money(v: float) -> str:
    a = abs(v)
    return f"${v / 1e6:,.2f}M" if a >= 1e6 else f"${v / 1e3:,.1f}K" if a >= 1e3 else f"${v:,.0f}"


def show(fig: go.Figure) -> None:
    """Transparent, theme-neutral Plotly figure (works in light & dark mode)."""
    grid = "rgba(128,128,128,.18)"
    fig.update_layout(margin=dict(l=8, r=8, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13, color="#8b92a5"),
                      title=dict(font=dict(size=16)), legend=dict(orientation="h", y=-0.16, title=None))
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displaylogo": False})


def table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True)


def kpi(label: str, value: str, d: float | None = None, invert: bool = False) -> str:
    if d is None:
        dh = '<div class="d flat">no prior period</div>'
    else:
        good = (d <= 0) if invert else (d >= 0)
        dh = (f'<div class="d {"up" if good else "down"}">{"▲" if d >= 0 else "▼"} {abs(d):.1%} '
              f'<span class="flat">vs previous period</span></div>')
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{dh}</div>'


def cards(items: list[tuple[str, str]], cls: str) -> None:
    html("".join(f'<div class="ins {cls}"><b class="h">{t}</b>{x}</div>' for t, x in items)
         or '<div class="ins">Nothing notable for this selection.</div>')


# ───────────────────────────── data layer ─────────────────────────────
def _dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return pd.to_datetime(s, format="%d/%m/%Y" if s.str.contains("/").any() else "%Y-%m-%d", errors="coerce")


@st.cache_data(show_spinner="Loading data…")
def load_data(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    if text.split("\n", 1)[0].split(",")[0].strip().strip('"').isdigit():   # headerless export
        df = pd.read_csv(io.StringIO(text), header=None, names=COLS, dtype={"Postal_Code": str})
    else:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    need = {"Order_ID", "Order_Date", "Ship_Date", "Ship_Mode", "Customer_ID", "Segment",
            "State", "Region", "Category", "Sub_Category", "Product_Name", "Sales"}
    if need - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(need - set(df.columns))}")
    df["Order_Date"], df["Ship_Date"] = _dates(df["Order_Date"]), _dates(df["Ship_Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").round(4)        # removes float32 noise
    df = df.dropna(subset=["Order_Date", "Ship_Date", "Sales"])
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Row_ID"]).copy()
    for c in ["Ship_Mode", "Segment", "State", "Region", "Category", "Sub_Category"]:
        df[c] = df[c].astype(str).str.strip()
    df["Month"] = df["Order_Date"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["Order_Date"].dt.year
    df["Month_Num"] = df["Order_Date"].dt.month
    df["Ship_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
    df["Sales_Band"] = pd.cut(df["Sales"], [-np.inf, 50, 500, np.inf], labels=BAND_ORDER, right=False).astype(str)
    return df


def get_raw() -> tuple[bytes | None, str]:
    for f in DATA_FILES:
        p = Path(__file__).parent / f
        if p.exists():
            return p.read_bytes(), p.name
    st.info("📂 No data file found next to app.py - upload your sales CSV to begin.")
    up = st.file_uploader("Sales CSV", type="csv")
    return (up.getvalue(), up.name) if up else (None, "")


def metrics(d: pd.DataFrame) -> dict:
    rev, n = d["Sales"].sum(), d["Order_ID"].nunique()
    return dict(rev=rev, orders=n, aov=rev / n if n else 0, cust=d["Customer_ID"].nunique(),
                ship=d["Ship_Days"].mean() if len(d) else 0)


def chg(cur: float, prev: float) -> float | None:
    return (cur - prev) / prev if prev else None


# ───────────────────────────── storytelling engine ─────────────────────────────
def build_story(f: pd.DataFrame, start, end, m: dict, prev_rev_delta: float | None):
    tot, good, warn, act = m["rev"], [], [], []
    cat = f.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    sub = f.groupby("Sub_Category")["Sales"].sum().sort_values(ascending=False)
    reg = f.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    sta = f.groupby("State")["Sales"].sum().sort_values(ascending=False)
    top_cat, top3 = cat.index[0], sub.head(3)

    good.append(("Category leader", f"<b>{top_cat}</b> delivers {cat.iloc[0] / tot:.0%} of revenue"
                 + (f", ahead of {cat.index[1]} ({cat.iloc[1] / tot:.0%})." if len(cat) > 1 else ".")))
    good.append(("Hero sub-categories", f"{', '.join(top3.index)} together generate {top3.sum() / tot:.0%} of revenue."))
    act.append(("Double down on winners", f"Protect stock, pricing and promotion budget for {', '.join(top3.index)} "
                f"- the {len(top3)} lines that carry {top3.sum() / tot:.0%} of sales."))

    yr = f.groupby("Year")["Sales"].sum()
    done = [y for y in yr.index
            if pd.Timestamp(start) <= pd.Timestamp(y, 1, 10) and pd.Timestamp(end) >= pd.Timestamp(y, 12, 20)]
    if len(done) >= 2 and done[-1] - done[-2] == 1:
        g = yr[done[-1]] / yr[done[-2]] - 1
        (good if g >= 0 else warn).append(("Year-over-year", f"{done[-1]} revenue {'grew' if g >= 0 else 'fell'} "
            f"{abs(g):.0%} vs {done[-2]} ({money(yr[done[-2]])} → {money(yr[done[-1]])})."))

    if (end - start).days >= 365:
        ms = f.groupby("Month_Num")["Sales"].sum().reindex(range(1, 13), fill_value=0)
        peak, low = sorted(ms.nlargest(3).index), ms.idxmin()
        pk = ", ".join(MONTHS[i - 1] for i in peak)
        good.append(("Seasonal peak", f"{pk} bring {ms[peak].sum() / tot:.0%} of revenue; "
                     f"{MONTHS[ms.idxmax() - 1]} is the single strongest month."))
        warn.append(("Soft season", f"{MONTHS[low - 1]} is the weakest month at only {ms[low] / tot:.1%} of revenue."))
        act.append(("Plan around seasonality", f"Front-load inventory, staffing and campaigns before {pk}; "
                    f"use {MONTHS[low - 1]} for clearance, B2B bundles and loyalty offers."))

    if len(reg) > 1:
        w = reg.index[-1]
        warn.append(("Regional bottleneck", f"<b>{w}</b> adds just {reg.iloc[-1] / tot:.0%} of revenue - "
                     f"{1 - reg.iloc[-1] / reg.mean():.0%} below the average region ({reg.index[0]} leads with {reg.iloc[0] / tot:.0%})."))
        act.append((f"Close the {w} gap", f"Set a catch-up target toward the regional average, review pricing and "
                    f"fulfilment there, and replicate what works in {reg.index[0]}."))
    if len(sta) > 3:
        warn.append(("Geographic concentration", f"Top 3 states ({', '.join(sta.index[:3])}) = "
                     f"{sta.head(3).sum() / tot:.0%} of revenue - a local shock would hurt."))

    g = f.groupby("Category").agg(L=("Sales", "size"), S=("Sales", "sum"))
    g["gap"] = g["L"] / g["L"].sum() - g["S"] / g["S"].sum()
    if len(g) > 1 and g["gap"].max() > 0.10:
        c = g["gap"].idxmax()
        warn.append(("Volume ≠ value", f"<b>{c}</b> is {g.loc[c, 'L'] / g['L'].sum():.0%} of line items but only "
                     f"{g.loc[c, 'S'] / g['S'].sum():.0%} of revenue - lots of work for small baskets."))
        act.append(("Lift basket value", f"Bundle and cross-sell {c} with {top_cat} items at checkout to raise average order value."))

    hi = f[f["Sales"] >= 500]
    if len(hi):
        warn.append(("High-ticket dependence", f"Lines of $500+ are {len(hi) / len(f):.0%} of items but "
                     f"{hi['Sales'].sum() / tot:.0%} of revenue - sales are driven by a few big deals."))
        act.append(("Run a key-account playbook", "Fast-track pricing approvals, stock reservation and delivery SLAs for big-ticket orders."))

    cu = f.groupby("Customer_ID")["Sales"].sum().sort_values(ascending=False)
    k = max(1, round(len(cu) * 0.2))
    good.append(("Customer base", f"{m['cust']:,} active customers; the top 20% ({k:,}) generate {cu.head(k).sum() / tot:.0%} of revenue."))
    act.append(("Protect top customers", "Launch a tiered loyalty / account-management programme for the top-20% group and win-back for lapsed buyers."))

    sh = f.groupby("Ship_Mode")["Sales"].sum().sort_values(ascending=False)
    if len(sh) > 1:
        warn.append(("Shipping mix", f"{sh.index[0]} carries {sh.iloc[0] / tot:.0%} of revenue; average delivery takes {m['ship']:.1f} days."))
        act.append(("Monetise speed", "Offer First Class / Same Day upgrades on high-value orders to lift margin and satisfaction."))

    seg = f.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
    lead = f"{top_cat} is the engine ({cat.iloc[0] / tot:.0%})"
    head = (f"Between <b>{start:%b %Y}</b> and <b>{end:%b %Y}</b> the business earned <b>{money(tot)}</b> from "
            f"<b>{m['orders']:,}</b> orders (average order <b>{money(m['aov'])}</b>). {lead}, "
            + (f"<b>{reg.index[0]}</b> is the top region ({reg.iloc[0] / tot:.0%}) and <b>{seg.index[0]}</b> the top segment ({seg.iloc[0] / tot:.0%}). " if len(reg) else "")
            + (f"Momentum is <b>{'up' if prev_rev_delta >= 0 else 'down'} {abs(prev_rev_delta):.0%}</b> versus the previous period."
               if prev_rev_delta is not None else ""))
    return head, good, warn, act


# ───────────────────────────── charts ─────────────────────────────
def trend_charts(f: pd.DataFrame) -> None:
    m = f.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
    full = pd.date_range(m["Month"].min(), m["Month"].max(), freq="MS")
    m = m.set_index("Month").reindex(full, fill_value=0).rename_axis("Month").reset_index()
    m["MA3"] = m["Sales"].rolling(3, min_periods=1).mean()
    m["MoM"] = (m["Sales"] / m["Sales"].shift(1) - 1).replace([np.inf, -np.inf], np.nan) * 100

    fig = go.Figure()
    fig.add_bar(x=m["Month"], y=m["Sales"], name="Monthly revenue", marker_color=INDIGO, opacity=.85,
                hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=m["Month"], y=m["MA3"], name="3-month average", mode="lines",
                    line=dict(color=AMBER, width=3, shape="spline"))
    fig.update_layout(title="Monthly revenue & 3-month moving average", hovermode="x unified", yaxis_tickprefix="$")
    show(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=m["Month"], y=m["MoM"], marker_color=[GREEN if (v == v and v >= 0) else ROSE for v in m["MoM"]],
                               hovertemplate="%{x|%b %Y}<br>%{y:+.1f}%<extra></extra>"))
        fig.update_layout(title="Month-over-month growth", yaxis_ticksuffix="%")
        show(fig)
    y = f.groupby(["Year", "Month_Num"], as_index=False)["Sales"].sum()
    with c2:
        yl = y.assign(Year=y["Year"].astype(str))
        fig = px.line(yl, x="Month_Num", y="Sales", color="Year", markers=True,
                      color_discrete_sequence=[INDIGO, TEAL, AMBER, ROSE, SLATE])
        fig.update_layout(title="Seasonality - year over year", yaxis_tickprefix="$")
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTHS, title=None)
        show(fig)

    pv = y.pivot(index="Year", columns="Month_Num", values="Sales").reindex(columns=range(1, 13))
    pv.index, pv.columns = pv.index.astype(str), MONTHS
    fig = px.imshow(pv, aspect="auto", color_continuous_scale="Purples", text_auto=".2s")
    fig.update_layout(title="Revenue heatmap - year × month", coloraxis_showscale=False)
    show(fig)


def product_charts(f: pd.DataFrame) -> None:
    cs = f.groupby(["Category", "Sub_Category"], as_index=False)["Sales"].sum()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.treemap(cs, path=["Category", "Sub_Category"], values="Sales", color="Category", color_discrete_map=CAT_COLORS)
        fig.update_traces(texttemplate="<b>%{label}</b><br>$%{value:,.0f}", textfont_size=14)
        fig.update_layout(title="Revenue map - category → sub-category", margin=dict(t=56))
        show(fig)
    with c2:
        fig = px.bar(cs.sort_values("Sales"), x="Sales", y="Sub_Category", color="Category", orientation="h",
                     color_discrete_map=CAT_COLORS, text_auto=".2s")
        fig.update_layout(title="Sub-category ranking", xaxis_tickprefix="$", yaxis_title=None)
        show(fig)

    sc = f.groupby("Sub_Category")["Sales"].sum().sort_values(ascending=False)
    cum = sc.cumsum() / sc.sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=sc.index, y=sc.values, name="Revenue", marker_color=INDIGO)
    fig.add_scatter(x=sc.index, y=cum.values, name="Cumulative %", mode="lines+markers",
                    line=dict(color=AMBER, width=3), secondary_y=True)
    fig.add_scatter(x=sc.index, y=[80] * len(sc), name="80% line", mode="lines",
                    line=dict(color=ROSE, dash="dot"), secondary_y=True)
    fig.update_yaxes(range=[0, 105], ticksuffix="%", showgrid=False, secondary_y=True)
    fig.update_layout(title="Pareto - how few sub-categories drive 80% of revenue")
    show(fig)

    c3, c4 = st.columns(2)
    g = f.groupby("Category").agg(L=("Sales", "size"), S=("Sales", "sum"))
    with c3:
        fig = go.Figure([go.Bar(x=g.index, y=g["L"] / g["L"].sum(), name="Share of line items", marker_color=TEAL),
                         go.Bar(x=g.index, y=g["S"] / g["S"].sum(), name="Share of revenue", marker_color=INDIGO)])
        fig.update_layout(title="Volume vs value by category", barmode="group", yaxis_tickformat=".0%")
        show(fig)
    b = f.groupby("Sales_Band").agg(L=("Sales", "size"), S=("Sales", "sum")).reindex(BAND_ORDER).fillna(0)
    with c4:
        fig = go.Figure([go.Bar(x=b.index, y=b["L"] / b["L"].sum(), name="Share of line items", marker_color=TEAL),
                         go.Bar(x=b.index, y=b["S"] / b["S"].sum(), name="Share of revenue", marker_color=INDIGO)])
        fig.update_layout(title="Sales distribution by ticket size", barmode="group", yaxis_tickformat=".0%")
        show(fig)

    tp = f.groupby("Product_Name")["Sales"].sum().nlargest(10).sort_values().reset_index()
    tp["Label"] = tp["Product_Name"].apply(lambda s: s if len(s) <= 44 else s[:43] + "…")
    fig = px.bar(tp, x="Sales", y="Label", orientation="h", text_auto=".2s", color_discrete_sequence=[INDIGO])
    fig.update_layout(title="Top 10 products by revenue", xaxis_tickprefix="$", yaxis_title=None)
    show(fig)


def geo_charts(f: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 1.4])
    r = f.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    with c1:
        fig = px.bar(r, x="Region", y="Sales", text_auto=".3s", color="Region",
                     color_discrete_sequence=[INDIGO, TEAL, AMBER, ROSE])
        fig.update_layout(title="Revenue by region", showlegend=False, yaxis_tickprefix="$")
        show(fig)
    s = f.groupby("State", as_index=False)["Sales"].sum()
    s["Abbr"] = s["State"].map(STATE_ABBR)
    with c2:
        fig = px.choropleth(s.dropna(subset=["Abbr"]), locations="Abbr", locationmode="USA-states", color="Sales",
                            scope="usa", hover_name="State", color_continuous_scale="Purples")
        fig.update_layout(title="Revenue by state", coloraxis_colorbar=dict(title="", tickprefix="$"),
                          geo=dict(bgcolor="rgba(0,0,0,0)"))
        show(fig)

    c3, c4 = st.columns(2)
    rc = f.groupby(["Region", "Category"], as_index=False)["Sales"].sum()
    with c3:
        fig = px.bar(rc, x="Region", y="Sales", color="Category", color_discrete_map=CAT_COLORS)
        fig.update_layout(title="Category mix by region", yaxis_tickprefix="$")
        show(fig)
    with c4:
        t = s.nlargest(10, "Sales").sort_values("Sales")
        fig = px.bar(t, x="Sales", y="State", orientation="h", text_auto=".2s", color_discrete_sequence=[TEAL])
        fig.update_layout(title="Top 10 states", xaxis_tickprefix="$", yaxis_title=None)
        show(fig)

    c5, c6, c7 = st.columns(3)
    seg = f.groupby("Segment").agg(S=("Sales", "sum"), O=("Order_ID", "nunique"))
    pal = [INDIGO, TEAL, AMBER, ROSE]
    with c5:
        fig = px.pie(seg.reset_index(), names="Segment", values="S", hole=.58, color_discrete_sequence=pal)
        fig.update_layout(title="Customer segments"); fig.update_traces(textinfo="percent+label", showlegend=False)
        show(fig)
    with c6:
        fig = px.pie(f, names="Ship_Mode", values="Sales", hole=.58, color_discrete_sequence=pal)
        fig.update_layout(title="Shipping mode"); fig.update_traces(textinfo="percent+label", showlegend=False)
        show(fig)
    with c7:
        a = (seg["S"] / seg["O"]).reset_index(name="AOV")
        fig = px.bar(a, x="Segment", y="AOV", text_auto="$.0f", color="Segment", color_discrete_sequence=pal)
        fig.update_layout(title="Average order value by segment", showlegend=False, yaxis_tickprefix="$")
        show(fig)


# ───────────────────────────── app ─────────────────────────────
def main() -> None:
    html('<div class="hero"><div class="t">📊 Superstore Sales Intelligence</div>'
         '<div class="s">Executive story · trends · products · geography - filter anything, instantly.</div></div>')
    raw, name = get_raw()
    if raw is None:
        st.stop()
    try:
        df = load_data(raw)
    except Exception as e:
        st.error(f"Could not read the data file: {e}")
        st.stop()

    dmin, dmax = df["Order_Date"].min().date(), df["Order_Date"].max().date()
    opts = {k: sorted(df[c].unique()) for k, c in [("f_reg", "Region"), ("f_cat", "Category"), ("f_seg", "Segment")]}

    def reset() -> None:
        st.session_state.update(f_dates=(dmin, dmax), **opts)

    if "f_dates" not in st.session_state:
        reset()

    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        st.button("↺ Reset all filters", on_click=reset)
        dates = st.date_input("Order date range", min_value=dmin, max_value=dmax, key="f_dates")
        sel_reg = st.multiselect("Region", opts["f_reg"], key="f_reg")
        sel_cat = st.multiselect("Category", opts["f_cat"], key="f_cat")
        sel_seg = st.multiselect("Customer segment", opts["f_seg"], key="f_seg")
        st.caption(f"Source: {name} · {len(df):,} rows · {dmin:%b %Y} → {dmax:%b %Y}")

    if isinstance(dates, (list, tuple)):
        start, end = (dates[0], dates[1]) if len(dates) == 2 else (dates[0], dates[0]) if dates else (dmin, dmax)
    else:
        start = end = dates
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)

    other = df["Region"].isin(sel_reg) & df["Category"].isin(sel_cat) & df["Segment"].isin(sel_seg)
    f = df[other & df["Order_Date"].between(s_ts, e_ts)]
    if f.empty:
        st.warning("No orders match the current filters - widen the date range or re-select options.")
        st.stop()

    span = (e_ts - s_ts).days + 1
    prev = df[other & df["Order_Date"].between(s_ts - pd.Timedelta(days=span), s_ts - pd.Timedelta(days=1))]
    m, pm = metrics(f), metrics(prev)
    has_prev = not prev.empty
    d = {k: chg(m[k], pm[k]) if has_prev else None for k in m}

    for col, args in zip(st.columns(5), [
        ("Total revenue", money(m["rev"]), d["rev"]), ("Orders", f"{m['orders']:,}", d["orders"]),
        ("Avg order value", money(m["aov"]), d["aov"]), ("Active customers", f"{m['cust']:,}", d["cust"]),
        ("Avg delivery time", f"{m['ship']:.1f} days", d["ship"], True)]):
        with col:
            html(kpi(*args))
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["📖 Executive Story", "📈 Trends", "🛍️ Products", "🌎 Regions & Segments", "🗂️ Data"])
    with t1:
        head, good, warn, act = build_story(f, start, end, m, d["rev"])
        html(f'<div class="story">💡 <b>The story in 30 seconds.</b> {head}</div>')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ✅ What's working"); cards(good, "good")
        with c2:
            st.markdown("#### ⚠️ Pain points"); cards(warn, "warn")
        with c3:
            st.markdown("#### 🎯 Recommended actions"); cards([(f"{i}. {t}", x) for i, (t, x) in enumerate(act, 1)], "act")
        st.caption("All insights are generated live from the filtered data - change a filter and the story rewrites itself.")
    with t2:
        trend_charts(f)
    with t3:
        product_charts(f)
    with t4:
        geo_charts(f)
    with t5:
        st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                           "filtered_sales.csv", "text/csv")
        table(f.sort_values("Sales", ascending=False).drop(columns=["Month", "Year", "Month_Num"]))
    st.caption("Built with Streamlit · Plotly · Pandas")


main()
