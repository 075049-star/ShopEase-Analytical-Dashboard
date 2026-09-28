"""
ShopEase Order Analytics - Streamlit Dashboard
Group ID: 018_041_049 | Fixed seed: 18041049

Run:  streamlit run app.py
Data: put shopease_cleaned.csv (preferred) or shopease_raw_orders.csv next to this file.
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

SEED = 18041049
SAMPLE_SIZE = 10001
NUMERIC_COLS = ["CustomerAge", "Quantity", "UnitPrice", "Discount", "Rating", "TotalAmount"]
CATEGORICAL_COLS = ["Gender", "City", "Category", "Product", "PaymentMethod", "OrderStatus"]

st.set_page_config(page_title="ShopEase Dashboard", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------- data ----
def parse_discount(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        return float(s.replace("%", "")) / 100 if s.endswith("%") else float(s)
    except ValueError:
        return np.nan


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning logic as the notebook (safe to run on already-clean data)."""
    df = raw.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    df["Gender"] = df["Gender"].str.lower().map(
        {"f": "Female", "female": "Female", "m": "Male", "male": "Male"}).fillna(df["Gender"])
    for c in ["City", "Category", "Product"]:
        df[c] = df[c].str.title()
    df["PaymentMethod"] = df["PaymentMethod"].str.lower().map(
        {"bank transfer": "Bank Transfer", "wallet": "Wallet", "cash": "Cash",
         "visa": "Visa", "mastercard": "Mastercard"}).fillna(df["PaymentMethod"])
    df["OrderStatus"] = df["OrderStatus"].str.lower().map(
        {"delivered": "Delivered", "pending": "Pending",
         "cancelled": "Cancelled", "canceled": "Cancelled"}).fillna(df["OrderStatus"])

    df["Discount"] = df["Discount"].apply(parse_discount)
    for c in ["CustomerAge", "Quantity", "UnitPrice", "Rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    if "DeliveryDate" in df.columns:
        df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

    df.loc[(df["CustomerAge"] < 10) | (df["CustomerAge"] > 100), "CustomerAge"] = np.nan
    df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
    df.loc[df["UnitPrice"] <= 0, "UnitPrice"] = np.nan
    df.loc[(df["Rating"] < 1) | (df["Rating"] > 5), "Rating"] = np.nan

    df["TotalAmount"] = (df["UnitPrice"] * df["Quantity"] * (1 - df["Discount"].fillna(0))).round(2)
    df = df.drop_duplicates().drop_duplicates(subset="OrderID", keep="first").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ["shopease_cleaned.csv", "shopease_raw_orders.csv"]:
        for base in (here, os.getcwd()):
            path = os.path.join(base, name)
            if os.path.exists(path):
                d = pd.read_csv(path)
                if name == "shopease_raw_orders.csv":
                    replace = len(d) < SAMPLE_SIZE
                    d = d.sample(n=SAMPLE_SIZE, random_state=SEED, replace=replace).reset_index(drop=True)
                return clean(d), name
    return None, None


df_all, source = load_data()

if df_all is None:
    st.error("No data file found. Put **shopease_cleaned.csv** or **shopease_raw_orders.csv** "
             "in the same folder as app.py and reload.")
    st.stop()

# -------------------------------------------------------------- sidebar ----
st.sidebar.title("🛒 Filters")
st.sidebar.caption(f"Source: {source} · n = {len(df_all):,}")


def multi(label, col):
    opts = sorted(df_all[col].dropna().unique())
    return st.sidebar.multiselect(label, opts, default=opts)


cats = multi("Category", "Category")
cities = multi("City", "City")
genders = multi("Gender", "Gender")
statuses = multi("Order status", "OrderStatus")

f = df_all[
    df_all["Category"].isin(cats) & df_all["City"].isin(cities)
    & df_all["Gender"].isin(genders) & df_all["OrderStatus"].isin(statuses)
]

if df_all["OrderDate"].notna().any():
    dmin, dmax = df_all["OrderDate"].min().date(), df_all["OrderDate"].max().date()
    dr = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr) == 2:
        f = f[f["OrderDate"].isna() | ((f["OrderDate"].dt.date >= dr[0]) & (f["OrderDate"].dt.date <= dr[1]))]

st.title("ShopEase Order Analytics Dashboard")
st.caption("Group 018_041_049 · Foundations of Big Data Analytics with Python")

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ----------------------------------------------------------------- KPIs ----
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Orders", f"{len(f):,}")
k2.metric("Revenue (EGP)", f"{f['TotalAmount'].sum():,.0f}")
k3.metric("Avg order value", f"{f['TotalAmount'].mean():,.0f}")
k4.metric("Avg rating", f"{f['Rating'].mean():.2f}")
k5.metric("Cancel rate", f"{(f['OrderStatus'] == 'Cancelled').mean() * 100:.1f}%")

tab_over, tab_desc, tab_viz, tab_inf, tab_reg, tab_data = st.tabs(
    ["📊 Overview", "🧮 Descriptive", "📈 Visualizations", "🧪 Inferential", "📉 Regression", "🗂 Data"]
)

# ------------------------------------------------------------- Overview ----
with tab_over:
    a, b = st.columns(2)
    rev_cat = f.groupby("Category")["TotalAmount"].sum().reset_index()
    a.plotly_chart(px.bar(rev_cat, x="Category", y="TotalAmount", color="Category",
                          title="Revenue by Category"), use_container_width=True)
    cnt_city = f["City"].value_counts().reset_index()
    cnt_city.columns = ["City", "Orders"]
    b.plotly_chart(px.bar(cnt_city, x="Orders", y="City", orientation="h",
                          title="Orders by City"), use_container_width=True)

    c, d = st.columns(2)
    weekly = (f.dropna(subset=["OrderDate"]).set_index("OrderDate")
                .resample("W")["TotalAmount"].sum().reset_index())
    c.plotly_chart(px.line(weekly, x="OrderDate", y="TotalAmount", markers=True,
                           title="Weekly Revenue Trend"), use_container_width=True)
    d.plotly_chart(px.pie(f, names="OrderStatus", title="Order Status Distribution", hole=0.35),
                   use_container_width=True)

# ---------------------------------------------------------- Descriptive ----
with tab_desc:
    st.subheader("Numeric variables")
    desc = f[NUMERIC_COLS].describe(percentiles=[.25, .5, .75]).T
    desc["skewness"] = f[NUMERIC_COLS].skew()
    desc["kurtosis"] = f[NUMERIC_COLS].kurt()
    desc["mode"] = f[NUMERIC_COLS].mode().iloc[0]
    st.dataframe(desc.round(2), use_container_width=True)

    st.subheader("Categorical variables")
    col = st.selectbox("Choose a variable", CATEGORICAL_COLS)
    vc = f[col].value_counts()
    table = pd.DataFrame({"Frequency": vc, "Relative Frequency": (vc / vc.sum()).round(4)})
    l, r = st.columns([1, 1])
    l.dataframe(table, use_container_width=True)
    r.plotly_chart(px.bar(table.reset_index(), x=table.index.name or "index", y="Frequency"),
                   use_container_width=True)
    st.caption(f"Highest: **{vc.idxmax()}** ({vc.max():,}) · Lowest: **{vc.idxmin()}** ({vc.min():,})")

# -------------------------------------------------------- Visualizations ----
with tab_viz:
    chart = st.selectbox("Chart type", [
        "Scatter", "Line (weekly revenue)", "Box-Whisker", "Violin", "Heat map (correlation)",
        "Histogram", "Bar", "Pie"])

    if chart == "Scatter":
        st.plotly_chart(px.scatter(f, x="UnitPrice", y="TotalAmount", color="Category", opacity=0.6,
                                   title="UnitPrice vs TotalAmount"), use_container_width=True)
    elif chart == "Line (weekly revenue)":
        w = (f.dropna(subset=["OrderDate"]).set_index("OrderDate")
               .resample("W")["TotalAmount"].sum().reset_index())
        st.plotly_chart(px.line(w, x="OrderDate", y="TotalAmount", markers=True), use_container_width=True)
    elif chart == "Box-Whisker":
        st.plotly_chart(px.box(f, x="Category", y="TotalAmount", points=False,
                               title="Order value by Category"), use_container_width=True)
    elif chart == "Violin":
        st.plotly_chart(px.violin(f, x="Category", y="Rating", box=True,
                                  title="Rating by Category"), use_container_width=True)
    elif chart == "Heat map (correlation)":
        st.plotly_chart(px.imshow(f[NUMERIC_COLS].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Pearson correlation"), use_container_width=True)
    elif chart == "Histogram":
        cap = f["TotalAmount"].quantile(0.99)
        st.plotly_chart(px.histogram(f[f["TotalAmount"] < cap], x="TotalAmount", nbins=40, marginal="box",
                                     title="Order value (below 99th percentile)"), use_container_width=True)
    elif chart == "Bar":
        var = st.selectbox("Variable", CATEGORICAL_COLS, key="bar_var")
        v = f[var].value_counts().reset_index()
        v.columns = [var, "Orders"]
        st.plotly_chart(px.bar(v, x=var, y="Orders", color=var), use_container_width=True)
    else:
        var = st.selectbox("Variable", CATEGORICAL_COLS, key="pie_var")
        st.plotly_chart(px.pie(f, names=var), use_container_width=True)

# ------------------------------------------------------------ Inferential ----
with tab_inf:
    st.subheader("95% confidence interval for mean TotalAmount")
    s = f["TotalAmount"].dropna()
    if len(s) > 2:
        ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
        st.write(f"Mean = **{s.mean():,.2f}** · 95% CI = **({ci[0]:,.2f}, {ci[1]:,.2f})**")

    st.subheader("Tests of mean")
    male = f.loc[f.Gender == "Male", "TotalAmount"].dropna()
    female = f.loc[f.Gender == "Female", "TotalAmount"].dropna()
    if len(male) > 1 and len(female) > 1:
        t, p = stats.ttest_ind(male, female, equal_var=False)
        st.write(f"Welch t-test (Male vs Female): t = {t:.3f}, p = {p:.4f} → "
                 f"{'significant' if p < 0.05 else 'not significant'}")
    else:
        st.info("t-test needs both Male and Female rows in the filter.")

    groups = [g.dropna().values for _, g in f.groupby("Category")["TotalAmount"] if g.notna().sum() > 1]
    if len(groups) >= 2:
        fs, pa = stats.f_oneway(*groups)
        st.write(f"One-way ANOVA (Category): F = {fs:.2f}, p = {pa:.6f} → "
                 f"{'significant' if pa < 0.05 else 'not significant'}")
        lv, lp = stats.levene(*groups)
        st.write(f"Levene's test of variance: W = {lv:.2f}, p = {lp:.6f}")

    st.subheader("Normality of TotalAmount")
    if len(s) >= 8:
        smp = s.sample(min(5000, len(s)), random_state=SEED)
        shw, shp = stats.shapiro(smp)
        ksd, ksp = stats.kstest(stats.zscore(smp), "norm")
        st.write(f"Shapiro-Wilk: W = {shw:.4f}, p = {shp:.6f}")
        st.write(f"Kolmogorov-Smirnov: D = {ksd:.4f}, p = {ksp:.6f}")

    st.subheader("Chi-square test of independence")
    c1 = st.selectbox("Variable 1", CATEGORICAL_COLS, index=2)
    c2 = st.selectbox("Variable 2", CATEGORICAL_COLS, index=5)
    if c1 != c2:
        ct = pd.crosstab(f[c1], f[c2])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            st.write(f"χ² = {chi2:.2f}, dof = {dof}, p = {p:.4f} → "
                     f"{'dependent' if p < 0.05 else 'independent (no evidence of association)'}")
            st.dataframe(ct, use_container_width=True)
    else:
        st.info("Pick two different variables.")

    st.subheader("Mann-Whitney U (Rating, Male vs Female)")
    mr = f.loc[f.Gender == "Male", "Rating"].dropna()
    fr = f.loc[f.Gender == "Female", "Rating"].dropna()
    if len(mr) > 1 and len(fr) > 1:
        u, up = stats.mannwhitneyu(mr, fr)
        st.write(f"U = {u:.1f}, p = {up:.4f} → {'significant' if up < 0.05 else 'not significant'}")

# ------------------------------------------------------------- Regression ----
with tab_reg:
    if not HAS_SM:
        st.error("statsmodels is not installed. Run: pip install statsmodels")
    else:
        st.subheader("Multiple linear regression")
        rd = f[["TotalAmount", "UnitPrice", "Quantity", "Discount", "Category"]].dropna()
        if len(rd) > 20 and rd["Category"].nunique() > 0:
            m = smf.ols("TotalAmount ~ UnitPrice + Quantity + Discount + C(Category)", data=rd).fit()
            r1, r2 = st.columns(2)
            r1.metric("R²", f"{m.rsquared:.3f}")
            r2.metric("Adj. R²", f"{m.rsquared_adj:.3f}")
            st.dataframe(pd.DataFrame({"coef": m.params, "p-value": m.pvalues}).round(4),
                         use_container_width=True)
            with st.expander("Full summary"):
                st.text(m.summary().as_text())

        st.subheader("Polynomial regression")
        pdf = f[["UnitPrice", "TotalAmount"]].dropna()
        deg = st.slider("Degree", 1, 5, 2)
        if len(pdf) > deg + 2:
            fn = np.poly1d(np.polyfit(pdf["UnitPrice"], pdf["TotalAmount"], deg))
            xs = np.linspace(pdf["UnitPrice"].min(), pdf["UnitPrice"].max(), 200)
            fig = px.scatter(pdf, x="UnitPrice", y="TotalAmount", opacity=0.3)
            fig.add_scatter(x=xs, y=fn(xs), mode="lines", name=f"Degree {deg}", line=dict(color="red"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Logistic regression - predicting cancellation")
        ld = f[["UnitPrice", "Quantity", "Discount", "OrderStatus"]].dropna().copy()
        ld["Cancelled"] = (ld["OrderStatus"] == "Cancelled").astype(int)
        if ld["Cancelled"].nunique() == 2 and len(ld) > 30:
            try:
                lm = sm.Logit(ld["Cancelled"], sm.add_constant(ld[["UnitPrice", "Quantity", "Discount"]])).fit(disp=0)
                st.write(f"Pseudo R² = **{lm.prsquared:.4f}**")
                st.dataframe(pd.DataFrame({"coef": lm.params, "p-value": lm.pvalues}).round(4),
                             use_container_width=True)
            except Exception as e:
                st.warning(f"Logistic model could not be fitted: {e}")
        else:
            st.info("Need both cancelled and non-cancelled orders in the filter.")

# ------------------------------------------------------------------- Data ----
with tab_data:
    st.dataframe(f, use_container_width=True, height=450)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "shopease_filtered.csv", "text/csv")
