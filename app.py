import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import utils as U

st.set_page_config(page_title="Bank Credit and GDP Growth in India", layout="wide")

BLUE, RED, GREEN, GREY, ORANGE = "#1f4e9c", "#c0392b", "#2e8b57", "#7f7f7f", "#e67e22"


@st.cache_data
def load_all():
    a = U.load_annual()
    s = pd.read_csv(U.DATA / "sectoral_credit.csv")
    n = pd.read_csv(U.DATA / "npa.csv")
    p = pd.read_csv(U.DATA / "population_group.csv")
    return a, s, n, p


A, SEC, NPA, POP = load_all()
GROWTH_COLS = ["credit_growth", "real_credit_growth_wpi", "real_credit_growth_cpi", "real_gdp_growth",
               "nominal_gdp_growth", "credit_gdp_gap", "deposit_growth"]
MEASURES = {"Nominal (merger-adjusted)": ("credit_growth", "wpi"),
            "Real, WPI-deflated": ("real_credit_growth_wpi", "wpi"),
            "Real, CPI-deflated": ("real_credit_growth_cpi", "cpi")}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
lo, hi = st.sidebar.slider("Years (financial year starting)", 1971, 2025, (1971, 2025))
measure = st.sidebar.radio("Credit growth measure", list(MEASURES), index=0)
CC, DEFL = MEASURES[measure]
no_covid = st.sidebar.checkbox("Mask the Covid year (2020-21)", value=False,
                               help="Sets 2020-21 growth values to missing so the outlier does not drive the statistics.")
st.sidebar.caption("Controls apply to the Overview, Relationship, Lead-lag and Episodes tabs.")

df = A[(A.fy >= lo) & (A.fy <= hi)].copy()
if no_covid:
    df.loc[df.fy == 2020, GROWTH_COLS] = np.nan
LABEL = {"credit_growth": "Credit growth (%)", "real_credit_growth_wpi": "Real credit growth, WPI (%)",
         "real_credit_growth_cpi": "Real credit growth, CPI (%)"}[CC]

st.title("Bank Credit Growth and GDP Growth in India")
st.caption("Scheduled commercial banks, 1971-72 to 2025-26. Source: RBI Handbook of Statistics on the Indian Economy 2025-26 "
           "and MoSPI national accounts.")

tabs = st.tabs(["Overview", "Relationship", "Lead-lag", "Episodes", "Composition and stress",
                "Geography of credit", "Simulator", "Data and notes"])

# ---------------------------------------------------------------- 1 overview
with tabs[0]:
    last, prev = A.iloc[-1], A.iloc[-2]
    c = st.columns(5)
    c[0].metric("Credit / GDP", f"{last.credit_to_gdp:.1f}%", f"{last.credit_to_gdp - prev.credit_to_gdp:+.1f} pp")
    c[1].metric("Credit growth", f"{last.credit_growth:.1f}%", f"{last.credit_growth - prev.credit_growth:+.1f} pp")
    c[2].metric("Real GDP growth", f"{last.real_gdp_growth:.2f}%", f"{last.real_gdp_growth - prev.real_gdp_growth:+.2f} pp")
    c[3].metric("Credit / deposits", f"{last.credit_to_deposit:.1f}%", f"{last.credit_to_deposit - prev.credit_to_deposit:+.1f} pp")
    c[4].metric("Latest year", last.year)

    fig = go.Figure()
    for i, (name, a0, b0) in enumerate(U.REGIMES):
        if i % 2 == 1 and b0 >= lo and a0 <= hi:
            fig.add_vrect(x0=max(a0, lo) - .5, x1=min(b0, hi) + .5, fillcolor="#eeeeee", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=df.fy, y=df[CC], name=LABEL, line=dict(color=BLUE, width=2.5), customdata=df.year,
                             hovertemplate="%{customdata}: %{y:.1f}%<extra>credit</extra>"))
    fig.add_trace(go.Scatter(x=df.fy, y=df.real_gdp_growth, name="Real GDP growth (%)", line=dict(color=RED, width=2.5),
                             customdata=df.year, hovertemplate="%{customdata}: %{y:.1f}%<extra>GDP</extra>"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Credit growth and real GDP growth (shaded bands mark alternate regimes)", height=420,
                      yaxis_title="per cent, year-on-year", legend=dict(orientation="h", y=1.1), margin=dict(t=80))
    st.plotly_chart(fig)

    l, r = st.columns(2)
    with l:
        show_dep = st.checkbox("Show deposits / GDP", value=True)
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=df.fy, y=df.credit_to_gdp, name="Credit / GDP", fill="tozeroy",
                                line=dict(color=BLUE), customdata=df.year, hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
        if show_dep:
            f2.add_trace(go.Scatter(x=df.fy, y=df.deposits_to_gdp, name="Deposits / GDP", line=dict(color=RED, dash="dash"),
                                    customdata=df.year, hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"))
        f2.update_layout(title="Financial deepening", yaxis_title="% of GDP", height=360, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(f2)
    with r:
        gap = df.credit_gdp_gap
        f3 = go.Figure(go.Bar(x=df.fy, y=gap, marker_color=np.where(gap >= 0, BLUE, RED), customdata=df.year,
                              hovertemplate="%{customdata}: %{y:.1f} pp<extra></extra>"))
        f3.update_layout(title="Credit growth minus nominal GDP growth (drives the credit/GDP ratio)", yaxis_title="percentage points", height=360)
        st.plotly_chart(f3)

# ---------------------------------------------------------------- 2 relationship
with tabs[1]:
    st.subheader("How strongly are credit growth and GDP growth related?")
    regs = [r[0] for r in U.REGIMES]
    chosen = st.multiselect("Regimes to include", regs, default=regs)
    d2 = df[df.regime.isin(chosen)]
    st_ = U.pearson(d2[CC], d2.real_gdp_growth)
    m = st.columns(5)
    m[0].metric("Observations", st_["n"])
    m[1].metric("Pearson r", f"{st_['r']:.2f}" if st_["n"] > 3 else "n/a", f"p = {st_['p']:.3f}" if st_["n"] > 3 else None, delta_color="off")
    m[2].metric("Spearman rho", f"{st_['rho']:.2f}" if st_["n"] > 3 else "n/a", f"p = {st_['prho']:.3f}" if st_["n"] > 3 else None, delta_color="off")
    m[3].metric("Slope (GDP pts per credit pt)", f"{st_['slope']:.3f}" if st_["n"] > 3 else "n/a")
    m[4].metric("R-squared", f"{st_['r'] ** 2:.2f}" if st_["n"] > 3 else "n/a")

    d2p = d2.dropna(subset=[CC, "real_gdp_growth"])
    fig = px.scatter(d2p, x=CC, y="real_gdp_growth", color="regime", hover_name="year", labels={CC: LABEL, "real_gdp_growth": "Real GDP growth (%)"},
                     height=430)
    if st_["n"] > 3:
        xs = np.linspace(d2p[CC].min(), d2p[CC].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=st_["intercept"] + st_["slope"] * xs, mode="lines", name="Fitted line",
                                 line=dict(color="black", dash="dash")))
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)

    l, r = st.columns(2)
    with l:
        w = st.slider("Rolling window (years)", 8, 25, 15)
        roll = df[CC].rolling(w, min_periods=int(w * .8)).corr(df.real_gdp_growth)
        f = go.Figure(go.Scatter(x=df.fy, y=roll, line=dict(color=BLUE, width=2.5), customdata=df.year,
                                 hovertemplate="window ending %{customdata}: %{y:.2f}<extra></extra>"))
        f.add_hline(y=0, line_color="black", line_width=1)
        f.update_layout(title=f"{w}-year rolling correlation", yaxis_title="correlation", yaxis_range=[-1, 1], height=360)
        st.plotly_chart(f)
    with r:
        t1, t2 = st.slider("Credit growth buckets (low / high cut-offs, %)", 5, 35, (10, 20))
        dd = df.dropna(subset=[CC, "real_gdp_growth"]).copy()
        dd["bucket"] = pd.cut(dd[CC], [-100, t1, t2, 1000], labels=[f"Below {t1}%", f"{t1}% to {t2}%", f"Above {t2}%"])
        g = dd.groupby("bucket", observed=False).real_gdp_growth.agg(["mean", "count"]).reset_index()
        f = go.Figure(go.Bar(x=g.bucket.astype(str), y=g["mean"], marker_color=[RED, GREY, BLUE],
                             text=[f"{v:.1f}% (n={n})" if n else "no years" for v, n in zip(g["mean"].fillna(0), g["count"])], textposition="outside"))
        f.update_layout(title="Average real GDP growth by level of credit growth", yaxis_title="real GDP growth (%)", height=360)
        st.plotly_chart(f)

    st.markdown("**Regime summary**")
    tab = df.groupby("regime").apply(lambda x: pd.Series({
        "Years": len(x), "Avg credit growth": x[CC].mean(), "Avg real GDP growth": x.real_gdp_growth.mean(),
        "Avg nominal GDP growth": x.nominal_gdp_growth.mean(), "Correlation": U.pearson(x[CC], x.real_gdp_growth)["r"],
        "Credit/GDP at end": x.credit_to_gdp.iloc[-1]}), include_groups=False).round(2)
    st.dataframe(tab)

# ---------------------------------------------------------------- 3 lead-lag
def render_leadlag():
    st.subheader("Does credit lead, lag or move with GDP?")
    cA, cB = st.columns(2)
    basis = cA.radio("Basis", ["Growth rates", "HP-filtered cycles (log real levels)"], horizontal=True)
    kmax = cB.slider("Maximum lead / lag (years)", 1, 6, 3)
    if basis == "Growth rates":
        cx, gy, nlab = df[CC].reset_index(drop=True), df.real_gdp_growth.reset_index(drop=True), "growth rates"
    else:
        lam = st.slider("HP smoothing parameter (6.25 is the standard for annual data)", 1.0, 100.0, 6.25)
        lv = U.real_levels(A, DEFL)
        lv = lv[(lv.fy >= lo) & (lv.fy <= hi)]
        if len(lv) < 15:
            st.warning("Select a longer year range (at least 15 years with the chosen deflator).")
            return
        cx = pd.Series(U.hp_filter(lv.ln_credit.values, lam)[1])
        gy = pd.Series(U.hp_filter(lv.ln_gdp.values, lam)[1])
        nlab = "HP cycles"
    cc_ = U.cross_corr(cx, gy, kmax)
    band = 1.96 / np.sqrt(cc_.n.max())
    colors = [BLUE if k > 0 else RED if k < 0 else GREY for k in cc_.k]
    f = go.Figure(go.Bar(x=cc_.k, y=cc_.r, marker_color=colors, customdata=np.c_[cc_.p, cc_.n],
                         hovertemplate="k=%{x}: r=%{y:.2f}, p=%{customdata[0]:.3f}, n=%{customdata[1]}<extra></extra>"))
    f.add_hline(y=band, line_dash="dash", line_color="black")
    f.add_hline(y=-band, line_dash="dash", line_color="black")
    f.update_layout(title=f"Correlation of credit at t-k with GDP at t ({nlab}); dashed lines = approx. 5% significance band",
                    xaxis_title="k   (negative: credit lags GDP | positive: credit leads GDP)", yaxis_title="correlation", height=400)
    st.plotly_chart(f)
    best = cc_.loc[cc_.r.idxmax()]
    word = "coincides with" if best.k == 0 else ("leads" if best.k > 0 else "lags")
    st.info(f"The strongest correlation is {best.r:.2f} at k = {int(best.k)}, i.e. the credit cycle {word} the GDP cycle "
            f"{'' if best.k == 0 else f'by about {abs(int(best.k))} year(s)'} in this selection.")

    st.markdown("**Granger causality (growth rates)**")
    p = st.slider("Number of lags", 1, 4, 1)
    rows = []
    for lab, x_, y_ in [("Credit does not cause GDP", df[CC], df.real_gdp_growth), ("GDP does not cause credit", df.real_gdp_growth, df[CC])]:
        F, pv, n = U.granger(y_.reset_index(drop=True), x_.reset_index(drop=True), p)
        rows.append({"Null hypothesis": lab, "F": round(F, 2), "p-value": round(pv, 3), "n": n,
                     "Reject at 10%?": "Yes" if pv < .10 else "No"})
    st.dataframe(pd.DataFrame(rows).set_index("Null hypothesis"))
    st.caption("Tests the predictive content of lags, not structural causation. Try masking 2020-21 in the sidebar.")


with tabs[2]:
    render_leadlag()

# ---------------------------------------------------------------- 4 episodes
with tabs[3]:
    st.subheader("Episodes of credit expansion and stress")
    names = list(U.EPISODES) + ["Custom range"]
    ep = st.selectbox("Episode", names)
    if ep == "Custom range":
        a0, b0 = st.slider("Custom range (financial year starting)", 1971, 2025, (2004, 2010))
        note = "Choose any window to compare average credit and GDP growth."
    else:
        a0, b0, note = U.EPISODES[ep]
    w_ = A[(A.fy >= a0) & (A.fy <= b0)]
    prior = A[A.fy == a0 - 1]
    cols = st.columns(5)
    cols[0].metric("Avg credit growth", f"{w_[CC].mean():.1f}%")
    cols[1].metric("Avg real GDP growth", f"{w_.real_gdp_growth.mean():.1f}%")
    cols[2].metric("Avg nominal GDP growth", f"{w_.nominal_gdp_growth.mean():.1f}%")
    base = prior.credit_to_gdp.iloc[0] if len(prior) else w_.credit_to_gdp.iloc[0]
    cols[3].metric("Credit/GDP at end", f"{w_.credit_to_gdp.iloc[-1]:.1f}%", f"{w_.credit_to_gdp.iloc[-1] - base:+.1f} pp over window", delta_color="off")
    cols[4].metric("Years", len(w_))
    st.write(note)
    ctx = A[(A.fy >= a0 - 4) & (A.fy <= b0 + 3)]
    f = go.Figure()
    f.add_vrect(x0=a0 - .5, x1=b0 + .5, fillcolor="#eeeeee", line_width=0, layer="below")
    f.add_trace(go.Scatter(x=ctx.fy, y=ctx[CC], name=LABEL, mode="lines+markers", line=dict(color=BLUE, width=2.5)))
    f.add_trace(go.Scatter(x=ctx.fy, y=ctx.real_gdp_growth, name="Real GDP growth (%)", mode="lines+markers", line=dict(color=RED, width=2.5)))
    f.add_hline(y=0, line_color="black", line_width=1)
    f.update_layout(title="Episode window (shaded) with surrounding years", height=380, yaxis_title="per cent", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f)
    st.dataframe(w_[["year", CC, "real_gdp_growth", "nominal_gdp_growth", "wpi_inflation", "credit_to_gdp", "credit_to_deposit"]].set_index("year").round(2))

# ---------------------------------------------------------------- 5 composition and stress
with tabs[4]:
    st.subheader("What is the credit financing? Investment, consumption and services")
    yrs = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    major = SEC[SEC.level == 1].copy()
    short = {"Agriculture and Allied Activities": "Agriculture", "Industry (Micro and Small, Medium and Large)": "Industry",
             "Services": "Services", "Personal Loans": "Personal loans"}
    major["name"] = major.sector.map(short)
    total = SEC[SEC.level == 0][yrs].iloc[0]
    sh = major.set_index("name")[yrs].div(total) * 100
    l, r = st.columns(2)
    with l:
        f = go.Figure()
        for nm, colr in zip(sh.index, [GREEN, BLUE, ORANGE, RED]):
            f.add_trace(go.Bar(x=yrs, y=sh.loc[nm], name=nm, marker_color=colr, hovertemplate="%{y:.1f}%<extra>" + nm + "</extra>"))
        f.update_layout(barmode="stack", title="Share of non-food bank credit", yaxis_title="%", height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f)
    with r:
        start = st.radio("Growth measured from", ["2021-22", "2022-23"], horizontal=True)
        n_years = 4 if start == "2021-22" else 3
        opts = SEC[SEC.level.isin([1, 2])].sector.tolist()
        default = ["Agriculture and Allied Activities", "Industry (Micro and Small, Medium and Large)", "Services", "Personal Loans",
                   "Non-Banking Financial Companies (NBFCs)", "Housing (Including Priority Sector Housing)", "Vehicle Loans", "Loan against gold jewellery"]
        pick = st.multiselect("Sectors", opts, default=default)
        sel = SEC[SEC.sector.isin(pick)].copy()
        sel["cagr"] = U.cagr(sel[start], sel["2025-26"], n_years)
        sel = sel.sort_values("cagr")
        f = go.Figure(go.Bar(x=sel.cagr, y=sel.sector.str.slice(0, 34), orientation="h", marker_color=BLUE,
                             text=[f"{v:.1f}%" for v in sel.cagr], textposition="outside"))
        tot = U.cagr(total[start], total["2025-26"], n_years)
        f.add_vline(x=tot, line_dash="dash", line_color="black", annotation_text=f"Non-food {tot:.1f}%", annotation_position="top")
        if start == "2022-23":
            for lab, a_, b_, colr in [("Real GFCF", 8453506, 10434799, RED), ("Real consumption", 14922828, 17993871, GREEN)]:
                f.add_vline(x=U.cagr(a_, b_, 3), line_dash="dot", line_color=colr, annotation_text=f"{lab} {U.cagr(a_, b_, 3):.1f}%", annotation_position="bottom")
        f.update_layout(title="Compound annual growth of outstanding credit", xaxis_title="% per year", height=400, margin=dict(l=10))
        st.plotly_chart(f)
    st.caption("Real GFCF and private consumption growth are from Table 4 (new 2022-23 base), so they are shown only when growth is measured from 2022-23. "
               "Loans against gold jewellery include a reclassification of agricultural loans by one bank from May 2024 (Table 42 note), so part of its growth is not new lending.")

    st.subheader("Bank stress and lending")
    cl, cr = st.columns([1, 1])
    grp = cl.selectbox("Bank group", sorted(NPA.bank_group.unique()), index=sorted(NPA.bank_group.unique()).index("Scheduled Commercial Banks"))
    lag = cr.radio("Credit growth timing", ["Same year", "Following year"], horizontal=True)
    nn = NPA[NPA.bank_group == grp].merge(A[["year", "credit_growth", "real_gdp_growth", "fy"]], on="year")
    nn["credit_t"] = nn.credit_growth if lag == "Same year" else nn.credit_growth.shift(-1)
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=nn.fy, y=nn.gnpa_ratio, name="Gross NPA ratio (%)", marker_color="#bbbbbb"), secondary_y=False)
    f.add_trace(go.Scatter(x=nn.fy, y=nn.credit_growth, name="Credit growth, SCBs (%)", line=dict(color=BLUE, width=2.5), mode="lines+markers"), secondary_y=True)
    f.update_layout(title="Gross NPA ratio and credit growth", height=380, legend=dict(orientation="h", y=1.15))
    f.update_yaxes(title_text="NPA ratio (%)", secondary_y=False)
    f.update_yaxes(title_text="Credit growth (%)", secondary_y=True)
    st.plotly_chart(f)
    s_ = U.pearson(nn.gnpa_ratio, nn.credit_t)
    st.write(f"Correlation between the NPA ratio and credit growth ({lag.lower()}): **{s_['r']:.2f}** (n = {s_['n']}, p = {s_['p']:.3f}).")

# ---------------------------------------------------------------- 6 geography
with tabs[5]:
    st.subheader("Where is credit deployed? Population groups of bank centres")
    st.caption("Credit is by place of utilisation and deposits by place of collection (Table 47). Population groups: rural, semi-urban, urban, metropolitan.")
    cA, cB = st.columns(2)
    meas = cA.radio("Measure", ["Credit", "Deposits"], horizontal=True)
    yr = cB.slider("Year (31 March)", 1997, 2026, 2026)
    pm = POP[POP.measure == meas]
    snap = pm[pm.year == yr]
    l, r = st.columns(2)
    with l:
        f = px.treemap(snap, path=["population_group"], values="amount_cr", color="population_group",
                       color_discrete_sequence=[GREEN, ORANGE, BLUE, RED])
        f.update_traces(texttemplate="%{label}<br>%{percentRoot:.1%}")
        f.update_layout(title=f"{meas} by population group, {yr}", height=380, margin=dict(t=60, l=5, r=5, b=5))
        st.plotly_chart(f)
    with r:
        wide = pm.pivot(index="year", columns="population_group", values="amount_cr")
        shr = wide.div(wide.sum(axis=1), axis=0) * 100
        f = go.Figure()
        for g_, colr in zip(["Rural", "Semi-urban", "Urban", "Metropolitan"], [GREEN, ORANGE, BLUE, RED]):
            f.add_trace(go.Scatter(x=shr.index, y=shr[g_], name=g_, stackgroup="one", line=dict(color=colr), hovertemplate="%{y:.1f}%<extra>" + g_ + "</extra>"))
        f.add_vline(x=yr, line_dash="dash", line_color="black")
        f.update_layout(title=f"Share of {meas.lower()} over time", yaxis_title="%", height=380, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f)
    cr_ = POP[POP.measure == "Credit"].pivot(index="year", columns="population_group", values="amount_cr")
    dp_ = POP[POP.measure == "Deposits"].pivot(index="year", columns="population_group", values="amount_cr")
    cd = cr_ / dp_ * 100
    f = go.Figure()
    for g_, colr in zip(["Rural", "Semi-urban", "Urban", "Metropolitan"], [GREEN, ORANGE, BLUE, RED]):
        f.add_trace(go.Scatter(x=cd.index, y=cd[g_], name=g_, line=dict(color=colr, width=2.5)))
    f.add_vline(x=yr, line_dash="dash", line_color="black")
    f.update_layout(title="Credit-deposit ratio by population group (%)", yaxis_title="%", height=380, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(f)
    m = st.columns(4)
    for i, g_ in enumerate(["Rural", "Semi-urban", "Urban", "Metropolitan"]):
        m[i].metric(f"{g_} credit-deposit ratio, {yr}", f"{cd.loc[yr, g_]:.0f}%")

# ---------------------------------------------------------------- 7 simulator
with tabs[6]:
    st.subheader("Simulator")
    last_ratio, last_cd = float(A.credit_to_gdp.iloc[-1]), float(A.credit_to_deposit.iloc[-1])
    sa, sb = st.tabs(["Credit-to-GDP projection", "Credit response model (Monte Carlo)"])

    with sa:
        st.write("Credit-to-GDP rises whenever credit grows faster than nominal GDP: ratio(t) = ratio(t-1) x (1 + credit growth) / (1 + nominal GDP growth).")
        c1, c2, c3, c4 = st.columns(4)
        H = c1.slider("Horizon (years)", 3, 15, 10)
        gc = c2.slider("Credit growth (% a year)", 5.0, 30.0, 14.0, 0.5)
        gn = c3.slider("Nominal GDP growth (% a year)", 5.0, 15.0, 10.0, 0.5)
        gd = c4.slider("Deposit growth (% a year)", 5.0, 25.0, 11.0, 0.5)
        yrsH = np.arange(2025, 2025 + H + 1)
        ratio = last_ratio * ((1 + gc / 100) / (1 + gn / 100)) ** np.arange(H + 1)
        cdr = last_cd * ((1 + gc / 100) / (1 + gd / 100)) ** np.arange(H + 1)
        f = go.Figure()
        f.add_trace(go.Scatter(x=A.fy, y=A.credit_to_gdp, name="History", line=dict(color=BLUE, width=2.5)))
        f.add_trace(go.Scatter(x=yrsH, y=ratio, name="Projection", line=dict(color=RED, width=2.5, dash="dash")))
        f.update_layout(title="Credit / GDP (%)", height=380, yaxis_title="% of GDP", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f)
        m = st.columns(3)
        m[0].metric(f"Credit/GDP in {yrsH[-1]}", f"{ratio[-1]:.1f}%", f"{ratio[-1] - last_ratio:+.1f} pp vs today")
        m[1].metric(f"Credit/deposit ratio in {yrsH[-1]}", f"{cdr[-1]:.1f}%", f"{cdr[-1] - last_cd:+.1f} pp vs today")
        m[2].metric("Annual gap (credit minus GDP growth)", f"{gc - gn:+.1f} pp")
        if cdr[-1] > 100:
            st.warning("Credit would exceed deposits in this scenario, which would need non-deposit funding.")
        gcs, gns = np.arange(6, 25, 2.0), np.arange(6, 15.5, 1.0)
        z = np.array([[last_ratio * ((1 + a_ / 100) / (1 + b_ / 100)) ** H for b_ in gns] for a_ in gcs])
        hm = go.Figure(go.Heatmap(z=z, x=gns, y=gcs, colorscale="Blues", colorbar=dict(title="Credit/GDP %"),
                                  hovertemplate="GDP %{x}% | credit %{y}% -> %{z:.0f}%<extra></extra>"))
        hm.update_layout(title=f"Credit/GDP after {H} years for every combination", xaxis_title="Nominal GDP growth (%)",
                         yaxis_title="Credit growth (%)", height=420)
        st.plotly_chart(hm)

    with sb:
        st.write("Credit growth is modelled the way the data say it behaves: it responds to last year's real GDP growth and to inflation "
                 "(credit(t) = a + b x real GDP growth(t-1) + c x WPI inflation(t) + Covid dummy + error). "
                 "Scenarios for GDP are then run through the fitted model, with historical residuals resampled for uncertainty.")
        fit_from = st.slider("Fit the model on years from", 1981, 2005, 1981)
        d = A[A.fy >= fit_from][["fy", "credit_growth", "real_gdp_growth", "wpi_inflation"]].copy()
        d["gdp_lag"] = d.real_gdp_growth.shift(1)
        d["covid"] = (d.fy == 2020).astype(float)
        d = d.dropna()
        X = np.column_stack([np.ones(len(d)), d.gdp_lag, d.wpi_inflation, d.covid])
        beta, res, sigma = U.fit_ols(d.credit_growth.values, X)
        hist_sd = float(A[(A.fy >= 1991) & (A.fy != 2020)].real_gdp_growth.std())
        cf = st.columns(4)
        cf[0].metric("Constant", f"{beta[0]:.2f}")
        cf[1].metric("Response to last year's GDP growth", f"{beta[1]:.2f}")
        cf[2].metric("Response to WPI inflation", f"{beta[2]:.2f}")
        cf[3].metric("Residual std. dev.", f"{sigma:.2f}")
        a1, a2, a3, a4 = st.columns(4)
        H2 = a1.slider("Horizon (years)", 3, 12, 8, key="h2")
        base = a2.slider("Baseline real GDP growth (%)", 2.0, 10.0, 6.5, 0.1)
        vol = a3.slider("GDP growth volatility (std. dev.)", 0.0, 4.0, round(hist_sd, 1), 0.1)
        infl = a4.slider("Inflation (%)", 0.0, 10.0, 4.0, 0.1)
        b1, b2, b3, b4 = st.columns(4)
        sy = b1.slider("Shock year (0 = none)", 0, H2, 0)
        ss = b2.slider("Shock size (pp of GDP growth)", -10.0, 4.0, -4.0, 0.5)
        thr = b3.slider("Credit/GDP threshold (%)", 55, 100, 70)
        nsim = b4.slider("Simulations", 200, 5000, 1000, 200)
        rng = np.random.default_rng(42)
        g_prev = np.full(nsim, float(A.real_gdp_growth.iloc[-1]))
        ratio_s = np.full(nsim, last_ratio)
        CG, RT, GG = [], [], []
        for t in range(1, H2 + 1):
            g_t = base + (ss if t == sy else 0.0) + rng.normal(0, vol, nsim)
            pi_t = np.clip(infl + rng.normal(0, 1.5, nsim), -2, None)
            e = rng.choice(res, nsim)
            cg = beta[0] + beta[1] * g_prev + beta[2] * pi_t + e
            ng = ((1 + g_t / 100) * (1 + pi_t / 100) - 1) * 100
            ratio_s = ratio_s * (1 + cg / 100) / (1 + ng / 100)
            CG.append(cg); RT.append(ratio_s.copy()); GG.append(g_t); g_prev = g_t
        CG, RT, GG = map(np.array, (CG, RT, GG))
        xs = np.arange(2026, 2026 + H2)

        def fan(arr, title, ytitle, hist_x, hist_y):
            q = np.percentile(arr, [5, 25, 50, 75, 95], axis=1)
            fg = go.Figure()
            fg.add_trace(go.Scatter(x=hist_x, y=hist_y, name="History", line=dict(color=BLUE, width=2.5)))
            fg.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[q[4], q[0][::-1]], fill="toself", fillcolor="rgba(192,57,43,0.15)",
                                    line=dict(width=0), name="5th-95th percentile"))
            fg.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[q[3], q[1][::-1]], fill="toself", fillcolor="rgba(192,57,43,0.30)",
                                    line=dict(width=0), name="25th-75th percentile"))
            fg.add_trace(go.Scatter(x=xs, y=q[2], name="Median", line=dict(color=RED, width=2.5)))
            fg.update_layout(title=title, yaxis_title=ytitle, height=380, legend=dict(orientation="h", y=1.15))
            return fg
        l, r = st.columns(2)
        with l:
            st.plotly_chart(fan(CG, "Simulated credit growth (%)", "%", A.fy[A.fy >= 2005], A.credit_growth[A.fy >= 2005]))
        with r:
            st.plotly_chart(fan(RT, "Simulated credit / GDP (%)", "% of GDP", A.fy[A.fy >= 2005], A.credit_to_gdp[A.fy >= 2005]))
        mm = st.columns(3)
        mm[0].metric("Median credit growth, final year", f"{np.median(CG[-1]):.1f}%")
        mm[1].metric("Median credit/GDP, final year", f"{np.median(RT[-1]):.1f}%")
        mm[2].metric(f"Probability credit/GDP > {thr}%", f"{(RT[-1] > thr).mean() * 100:.0f}%")
        st.caption("This is a descriptive model of how credit has accommodated GDP, not a forecast and not a causal estimate of what credit does to GDP. "
                   "Inflation in the GDP deflator is assumed equal to the inflation input. Because the fitted constant reflects decades of financial deepening, the model keeps credit growing faster than GDP; "
                   "fitting from a later year shows how sensitive this is.")

# ---------------------------------------------------------------- 8 data and notes
with tabs[7]:
    st.subheader("Data")
    which = st.selectbox("Dataset", ["Annual credit and GDP", "Sectoral credit (Table 42)", "Gross NPAs (Table 50)", "Population groups (Table 47)"])
    data_map = {"Annual credit and GDP": A.drop(columns=["fy", "regime"]), "Sectoral credit (Table 42)": SEC,
                "Gross NPAs (Table 50)": NPA, "Population groups (Table 47)": POP}
    sel_df = data_map[which]
    st.dataframe(sel_df)
    st.download_button("Download CSV", sel_df.to_csv(index=False).encode(), file_name=which.split(" (")[0].lower().replace(" ", "_") + ".csv", mime="text/csv")
    st.markdown("""
**Sources.** RBI *Handbook of Statistics on the Indian Economy 2025-26*: Tables 232 and 41 (bank credit growth and levels), 233 (credit and deposits as % of GDP),
235 and 236 (WPI and CPI inflation), 227 (GDP growth from 2019-20), 42 (sectoral deployment of credit), 50 (gross NPAs), 47 (population-group data) and 4 (GDP components).
Real and nominal GDP growth before 2022-23 are from MoSPI national accounts (2011-12 base).

**Adjustments.** Credit growth for 2023-24 (16.3%) and 2024-25 (12.1%) is adjusted for the July 2023 bank merger. Real credit growth is nominal growth deflated by WPI or CPI,
not by the GDP deflator, so that GDP does not appear on both sides of a comparison. The 2022-23 onward GDP series is on the new base year.
2025-26 figures are provisional.
""")
