import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. MODERN PASTEL STYLING & STRICT VIEWPORT LOCKING
PASTEL_COLORS = [
    "#38BDF8",  # Sky Blue
    "#818CF8",  # Indigo Accent
    "#A78BFA",  # Soft Purple
    "#34D399",  # Emerald Green
    "#F472B6",  # Pink Accent
]


def apply_dark_chart_theme(fig):
    """Consistent dark background with ultra-compact padding for tight fitting."""
    fig.update_layout(
        paper_bgcolor="#151B34",
        plot_bgcolor="#151B34",
        font=dict(color="#F8FAFC", size=9),
        legend=dict(font=dict(color="#F8FAFC", size=9), margin=dict(l=0, r=0, t=0, b=0)),
        margin=dict(l=5, r=5, t=18, b=5),
        autosize=True,
    )
    fig.update_xaxes(
        gridcolor="#232D52",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1", size=8),
        title_font=dict(color="#F8FAFC", size=9),
        title_standoff=2,
    )
    fig.update_yaxes(
        gridcolor="#232D52",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1", size=8),
        title_font=dict(color="#F8FAFC", size=9),
        title_standoff=2,
    )
    return fig


st.markdown(
    """
<style>
    /* Lock Viewport to 100vh and Disable Global Scrolling */
    html, body, [data-testid="stAppViewContainer"], .main {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    .stApp {
        background-color: #0B1020;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        height: 100vh !important;
        overflow: hidden !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], .stAppHeader {
        background-color: #080C18 !important;
        height: 2.5rem !important;
    }

    [data-testid="stMainBlockContainer"], .block-container {
        background-color: #0B1020;
        max-width: none !important;
        padding: 0.2rem 1rem 0.2rem !important;
        height: calc(100vh - 2.5rem) !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }

    /* Ultra-Compact Header Banner */
    .header-container {
        background: #111827;
        border: 1px solid #232D52;
        padding: 6px 14px;
        border-radius: 8px;
        color: #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 50px;
    }

    .header-title-box h1 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .header-title-box p {
        margin: 0;
        font-size: 0.72rem;
        color: #94A3B8;
    }

    .header-badge-box {
        text-align: right;
        background: #1E293B;
        border: 1px solid #334155;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .header-badge-title {
        font-size: 0.58rem;
        font-weight: 700;
        color: #818CF8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .header-badge-value {
        font-size: 0.72rem;
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Single-Line Compact KPI Cards */
    .kpi-card {
        background-color: #151B34;
        border-radius: 8px;
        padding: 6px 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: 1px solid #232D52;
        border-top: 3px solid #818CF8;
        height: 52px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-label {
        font-size: 0.62rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .kpi-value {
        font-size: 1.05rem;
        line-height: 1.1;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 1px;
    }

    /* Compact Headings & Containers */
    .section-label {
        margin: 2px 0 2px;
        color: #E2E8F0;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .card-title {
        margin: 0 0 2px;
        color: #F8FAFC;
        font-size: 0.8rem;
        font-weight: 700;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #151B34;
        border: 1px solid #232D52 !important;
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.4rem 0.5rem !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0F1530;
        border-right: 1px solid #232D52;
        min-width: 220px;
        max-width: 240px;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.5rem;
    }

    /* Force DataFrame height and compact row scaling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #232D52;
        border-radius: 6px;
        max-height: 100px !important;
    }

    div[data-testid="stDataFrame"] div[role="grid"] {
        max-height: 100px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="header-container">
        <div class="header-title-box">
            <h1>Customer Analytics & Segmentation</h1>
            <p>Explore customer behavior clusters, analyze key business metrics, and predict real-time customer segments.</p>
        </div>
        <div class="header-badge-box">
            <div class="header-badge-title">DATA PERIOD</div>
            <div class="header-badge-value">2010 – 2011</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# 3. DATA & MODEL LOADING
@st.cache_data
def load_data():
    data_path = os.path.join("data", "raw", "Online_Retail.csv")
    transactions = pd.read_csv(data_path)

    transactions["CustomerID"] = transactions["CustomerID"].fillna(
        "guest_" + transactions["InvoiceNo"].astype(str)
    )
    transactions["CustomerID"] = transactions["CustomerID"].astype(str)
    transactions = transactions[
        ~transactions["InvoiceNo"].astype(str).str.startswith("C")
        & (transactions["Quantity"] > 0)
        & (transactions["UnitPrice"] > 0)
    ].copy()

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], format="%m/%d/%Y %H:%M"
    )
    transactions["TotalAmount"] = (
        transactions["Quantity"] * transactions["UnitPrice"]
    )

    customer_transactions = transactions[
        ~transactions["CustomerID"].str.startswith("guest_")
    ].copy()
    cutoff_date = customer_transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    recency_df = (
        customer_transactions.groupby("CustomerID")["InvoiceDate"]
        .max()
        .reset_index()
    )
    recency_df["Recency"] = (
        cutoff_date - recency_df["InvoiceDate"]
    ).dt.days

    fm_df = (
        customer_transactions.groupby("CustomerID")
        .agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    rfm = recency_df[["CustomerID", "Recency"]].merge(fm_df, on="CustomerID")
    rfm["Frequency_Log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_Log"] = np.log1p(rfm["Monetary"])
    rfm["Recency_Segment"] = pd.cut(
        rfm["Recency"],
        bins=[-1, 30, 90, 180, 365, 1000],
        labels=[
            "Active (0-30d)",
            "Warm (31-90d)",
            "Cold (91-180d)",
            "At Risk (181-365d)",
            "Lapsed (365d+)",
        ],
    )
    return transactions, rfm


@st.cache_resource
def load_models():
    scaler_path = os.path.join("notebooks", "models", "scaler.joblib")
    model_path = os.path.join("notebooks", "models", "kmeans_model.joblib")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


transactions, df = load_data()
scaler, kmeans = load_models()

df["Cluster"] = kmeans.predict(
    scaler.transform(df[["Frequency_Log", "Monetary_Log"]])
)
segment_map = {
    0: "Lapsed / Low-Value",
    1: "Champions / High-Value",
    2: "Occasional / Mid-Value",
}
df["Segment_Name"] = df["Cluster"].map(segment_map)

# 4. SIDEBAR FILTERS
st.sidebar.markdown("### Filter Panel")
countries = ["All Countries"] + sorted(transactions["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Filter sales by Country", countries)
sales_df = transactions.copy()
if selected_country != "All Countries":
    sales_df = sales_df[sales_df["Country"] == selected_country]

segments = ["All Segments"] + list(df["Segment_Name"].dropna().unique())
selected_segment = st.sidebar.selectbox("Filter customer insights by Segment", segments)
if selected_segment != "All Segments":
    df = df[df["Segment_Name"] == selected_segment]

# 5. DASHBOARD TABS
tab1, tab2 = st.tabs(["Overview & Visualizations", "Real-time Predictor"])

with tab1:
    total_revenue = sales_df["TotalAmount"].sum()
    total_orders = sales_df["InvoiceNo"].nunique()
    average_order_value = total_revenue / total_orders if total_orders else 0
    total_units = sales_df["Quantity"].sum()
    registered_orders = sales_df.loc[~sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
    guest_orders = sales_df.loc[sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
    registered_share = registered_orders / total_orders if total_orders else 0
    guest_share = guest_orders / total_orders if total_orders else 0

    # 1-Row Compact KPIs
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    with k1:
        st.markdown(f'<div class="kpi-card" style="border-top-color: #38BDF8;"><div class="kpi-label">Total Revenue</div><div class="kpi-value">${total_revenue/1e6:.2f}M</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card" style="border-top-color: #818CF8;"><div class="kpi-label">Total Orders</div><div class="kpi-value">{total_orders:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card" style="border-top-color: #A78BFA;"><div class="kpi-label">Avg Order Value</div><div class="kpi-value">${average_order_value:,.2f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card" style="border-top-color: #34D399;"><div class="kpi-label">Units Sold</div><div class="kpi-value">{total_units/1e6:.2f}M</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card" style="border-top-color: #F472B6;"><div class="kpi-label">Reg / Guest</div><div class="kpi-value">{registered_share:.0%} / {guest_share:.0%}</div></div>', unsafe_allow_html=True)

    # Main Grid (Fits into Viewport Height without Scrolling)
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
            monthly_revenue = (
                sales_df.assign(Month=sales_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
                .groupby("Month", as_index=False)["TotalAmount"].sum()
            )
            fig_monthly = px.line(monthly_revenue, x="Month", y="TotalAmount", markers=True, template="plotly_dark")
            fig_monthly.update_traces(line_color=PASTEL_COLORS[0], marker=dict(size=4))
            fig_monthly.update_layout(height=160, hovermode="x unified")
            apply_dark_chart_theme(fig_monthly)
            st.plotly_chart(fig_monthly, use_container_width=True)

        with st.container(border=True):
            st.markdown('<div class="card-title">Revenue by Country</div>', unsafe_allow_html=True)
            country_revenue = transactions.groupby("Country", as_index=False)["TotalAmount"].sum().nlargest(5, "TotalAmount").sort_values("TotalAmount")
            fig_country = px.bar(country_revenue, x="TotalAmount", y="Country", orientation="h", color="Country", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark")
            fig_country.update_layout(showlegend=False, height=160, bargap=0.2)
            apply_dark_chart_theme(fig_country)
            st.plotly_chart(fig_country, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="card-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
            top_products = sales_df.groupby("Description", as_index=False)["TotalAmount"].sum().nlargest(10, "TotalAmount").sort_values("TotalAmount")
            top_products["DisplayDescription"] = top_products["Description"].str.slice(0, 22).where(
                top_products["Description"].str.len() <= 22, top_products["Description"].str.slice(0, 19) + "..."
            )
            fig_products = px.bar(top_products, x="TotalAmount", y="DisplayDescription", orientation="h", template="plotly_dark", color_discrete_sequence=[PASTEL_COLORS[1]])
            fig_products.update_layout(height=360, margin=dict(l=5, r=5, t=10, b=5))
            apply_dark_chart_theme(fig_products)
            fig_products.update_yaxes(title_text="", tickfont=dict(size=8))
            st.plotly_chart(fig_products, use_container_width=True)

    with col3:
        with st.container(border=True):
            st.markdown('<div class="card-title">Revenue by Customer Segment</div>', unsafe_allow_html=True)
            segment_summary = df.groupby("Segment_Name", as_index=False).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"), Avg_Frequency=("Frequency", "mean")).sort_values("Revenue", ascending=False)
            
            # Explicitly shrink data table height & rows
            segment_table = (
                segment_summary.style
                .format({"Revenue": "${:,.0f}", "Avg_Frequency": "{:.1f}"})
                .set_properties(**{"background-color": "#151B34", "color": "#F8FAFC", "font-size": "10px", "padding": "2px 4px"})
                .set_table_styles([{"selector": "th", "props": [("background-color", "#232D52"), ("color", "#FFFFFF"), ("font-size", "10px"), ("padding", "2px 4px")]}])
            )
            st.dataframe(segment_table, hide_index=True, use_container_width=True, height=95)

        with st.container(border=True):
            st.markdown('<div class="card-title">Customer Recency / Retention</div>', unsafe_allow_html=True)
            recency_order = ["Active (0-30d)", "Warm (31-90d)", "Cold (91-180d)", "At Risk (181-365d)", "Lapsed (365d+)"]
            recency_summary = df["Recency_Segment"].value_counts().reindex(recency_order, fill_value=0).rename_axis("Recency Segment").reset_index(name="Customers")
            fig_recency = px.bar(recency_summary, x="Recency Segment", y="Customers", color="Recency Segment", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark")
            fig_recency.update_layout(showlegend=False, height=180, bargap=0.2)
            apply_dark_chart_theme(fig_recency)
            fig_recency.update_xaxes(title_text="", tickfont=dict(size=7))
            fig_recency.update_yaxes(title_text="")
            st.plotly_chart(fig_recency, use_container_width=True)

with tab2:
    st.subheader("Real-Time Customer Segment Predictor")
    with st.form("rfm_predict_form"):
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            input_recency = st.number_input("Recency (Days)", min_value=0, value=10)
        with cp2:
            input_frequency = st.number_input("Frequency (Orders)", min_value=1, value=5)
        with cp3:
            input_monetary = st.number_input("Monetary Value ($)", min_value=0.0, value=250.0)
        submit_button = st.form_submit_button("Classify Customer")

    if submit_button and scaler is not None and kmeans is not None:
        freq_log = np.log1p(input_frequency)
        monetary_log = np.log1p(input_monetary)
        user_input_scaled = scaler.transform(np.array([[freq_log, monetary_log]]))
        predicted_cluster = kmeans.predict(user_input_scaled)[0]
        st.success(f"Assigned Segment: Cluster {predicted_cluster}")