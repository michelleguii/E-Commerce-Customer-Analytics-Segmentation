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

# 2. COLOR PALETTE & STYLING
PASTEL_COLORS = [
    "#38BDF8",  # Sky Blue
    "#818CF8",  # Indigo Accent
    "#A78BFA",  # Soft Purple
    "#34D399",  # Emerald Green
    "#F472B6",  # Pink Accent
]


def apply_dark_chart_theme(fig):
    """Consistent dark background, clean gridlines, and bold chart labels."""
    fig.update_layout(
        paper_bgcolor="#151B34",
        plot_bgcolor="#151B34",
        font=dict(color="#F8FAFC", size=11),
        legend=dict(font=dict(color="#F8FAFC", size=10)),
        margin=dict(l=10, r=10, t=28, b=10),
    )
    fig.update_xaxes(
        gridcolor="#232D52",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1", size=10),
        title_font=dict(color="#F8FAFC", size=11),
        title_standoff=6,
    )
    fig.update_yaxes(
        gridcolor="#232D52",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1", size=10),
        title_font=dict(color="#F8FAFC", size=11),
        title_standoff=6,
    )
    return fig


st.markdown(
    """
<style>
    /* Global Layout & Padding Fixes */
    .stApp {
        background-color: #0B1020;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], .stAppHeader {
        background-color: #080C18 !important;
    }

    [data-testid="stMainBlockContainer"], .block-container {
        background-color: #0B1020;
        max-width: none !important;
        padding: 1rem 1.25rem 0.75rem !important;
    }

    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.3rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }

    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp li, .stApp div[data-testid="stMarkdownContainer"] {
        color: #F8FAFC;
    }

    /* Enhanced Header Banner */
    .header-container {
        background: #111827;
        border: 1px solid #232D52;
        padding: 14px 20px;
        border-radius: 12px;
        color: #E2E8F0;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-title-box h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .header-title-box p {
        margin: 3px 0 0;
        font-size: 0.85rem;
        color: #94A3B8;
    }

    .header-badge-box {
        text-align: right;
        background: #1E293B;
        border: 1px solid #334155;
        padding: 6px 12px;
        border-radius: 8px;
    }

    .header-badge-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #818CF8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .header-badge-value {
        font-size: 0.82rem;
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Custom KPI Cards with Top-Border Accent */
    .kpi-card {
        background-color: #151B34;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.22);
        border: 1px solid #232D52;
        border-top: 3px solid #818CF8;
        min-height: 88px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .kpi-value {
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 800;
        color: #FFFFFF;
        margin: 4px 0 2px;
    }

    .kpi-subtext {
        font-size: 0.7rem;
        color: #38BDF8;
        font-weight: 500;
    }

    /* Section Headings & Card Titles */
    .section-label {
        margin: 6px 0 8px;
        color: #E2E8F0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .card-title {
        margin: 0 0 6px;
        color: #F8FAFC;
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    /* Insight Strip */
    .insight-strip {
        display: flex;
        gap: 8px;
        margin: 4px 0 10px;
        flex-wrap: wrap;
    }

    .insight-chip {
        background: #11182D;
        border: 1px solid #232D52;
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 0.74rem;
        color: #94A3B8;
    }

    .insight-chip strong {
        color: #38BDF8;
        font-weight: 600;
    }

    /* Container Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #151B34;
        border: 1px solid #232D52 !important;
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.18);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.75rem 0.85rem !important;
    }

    div[data-testid="stPlotlyChart"] {
        margin-top: -0.1rem;
    }

    div[data-testid="stCaptionContainer"] {
        font-size: 0.72rem;
        line-height: 1.25;
        color: #94A3B8;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0F1530;
        border-right: 1px solid #232D52;
        min-width: 240px;
        max-width: 260px;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.5rem 0.75rem;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #151B34 !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }

    div[data-baseweb="select"] * { color: #F8FAFC !important; }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg,
    [data-baseweb="popover"] *,
    [role="listbox"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #151B34 !important;
        color: #F8FAFC !important;
    }

    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #312E81 !important;
    }

    button[kind="secondary"], button[kind="primary"], div[data-testid="stFormSubmitButton"] > button {
        background-color: #4338CA !important;
        color: #FFFFFF !important;
        border: 1px solid #818CF8 !important;
    }

    button[kind="secondary"]:hover, button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #4F46E5 !important;
        border-color: #C4B5FD !important;
    }

    /* Forms & Tables */
    .stForm, div[data-testid="stForm"] {
        background-color: #151B34;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #232D52;
    }

    div[data-testid="stForm"] input {
        background-color: #0F1530 !important;
        color: #F8FAFC !important;
    }

    button[data-baseweb="tab"] {
        color: #F8FAFC !important;
        font-size: 0.88rem;
        padding: 0.45rem 0.9rem;
    }

    button[data-baseweb="tab"] p { margin: 0; }

    [data-testid="stTabs"] [data-testid="stTabsContent"] {
        padding-top: 0.4rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #232D52;
        border-radius: 8px;
        overflow: hidden;
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
st.sidebar.caption("Filter sales insights by country and customer insights by segment.")

countries = ["All Countries"] + sorted(transactions["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Filter sales by Country", countries)
sales_df = transactions.copy()
if selected_country != "All Countries":
    sales_df = sales_df[sales_df["Country"] == selected_country]

segments = ["All Segments"] + list(df["Segment_Name"].dropna().unique())
selected_segment = st.sidebar.selectbox("Filter customer insights by Segment", segments)
if selected_segment != "All Segments":
    df = df[df["Segment_Name"] == selected_segment]

st.sidebar.markdown("---")
st.sidebar.info("Tip: Use the prediction tool at the bottom of the page to classify new customer inputs.")

# 5. DASHBOARD TABS
tab1, tab2 = st.tabs(["Overview & Visualizations", "Real-time Predictor"])

with tab1:
    st.markdown('<div class="section-label">Performance Overview</div>', unsafe_allow_html=True)
    
    total_revenue = sales_df["TotalAmount"].sum()
    total_orders = sales_df["InvoiceNo"].nunique()
    average_order_value = total_revenue / total_orders if total_orders else 0
    total_units = sales_df["Quantity"].sum()
    registered_orders = sales_df.loc[~sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
    guest_orders = sales_df.loc[sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
    registered_share = registered_orders / total_orders if total_orders else 0
    guest_share = guest_orders / total_orders if total_orders else 0

    # Customized HTML KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    with k1:
        st.markdown(f'''
            <div class="kpi-card" style="border-top-color: #38BDF8;">
                <div class="kpi-label"> Total Revenue</div>
                <div class="kpi-value">${total_revenue/1e6:.2f}M</div>
                <div class="kpi-subtext">Gross Revenue</div>
            </div>
        ''', unsafe_allow_html=True)
    with k2:
        st.markdown(f'''
            <div class="kpi-card" style="border-top-color: #818CF8;">
                <div class="kpi-label"> Total Orders</div>
                <div class="kpi-value">{total_orders:,}</div>
                <div class="kpi-subtext">Completed Invoices</div>
            </div>
        ''', unsafe_allow_html=True)
    with k3:
        st.markdown(f'''
            <div class="kpi-card" style="border-top-color: #A78BFA;">
                <div class="kpi-label"> Avg Order Value</div>
                <div class="kpi-value">${average_order_value:,.2f}</div>
                <div class="kpi-subtext">Per Basket Size</div>
            </div>
        ''', unsafe_allow_html=True)
    with k4:
        st.markdown(f'''
            <div class="kpi-card" style="border-top-color: #34D399;">
                <div class="kpi-label"> Units Sold</div>
                <div class="kpi-value">{total_units/1e6:.2f}M</div>
                <div class="kpi-subtext">Items Volume</div>
            </div>
        ''', unsafe_allow_html=True)
    with k5:
        st.markdown(f'''
            <div class="kpi-card" style="border-top-color: #F472B6;">
                <div class="kpi-label"> Registered / Guest</div>
                <div class="kpi-value">{registered_share:.0%} / {guest_share:.0%}</div>
                <div class="kpi-subtext">User Distribution</div>
            </div>
        ''', unsafe_allow_html=True)

    st.caption("Sales metrics reflect the selected country. Customer segmentation uses registered customers only.")

    top_market = (
        sales_df.groupby("Country")["TotalAmount"].sum().idxmax()
        if not sales_df.empty else "N/A"
    )
    largest_segment = (
        df["Segment_Name"].value_counts().idxmax()
        if not df.empty else "N/A"
    )
    active_customers = (
        (df["Recency_Segment"] == "Active (0-30d)").mean()
        if not df.empty else 0
    )

    st.markdown(
        f'''
        <div class="insight-strip">
            <div class="insight-chip">Top market: <strong>{top_market}</strong></div>
            <div class="insight-chip">Largest customer segment: <strong>{largest_segment}</strong></div>
            <div class="insight-chip">Active customers: <strong>{active_customers:.1%}</strong></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Row 1 Charts (Adjusted Heights for Visual Balance)
    col_left, col_right = st.columns([3, 2], gap="small")
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
            monthly_revenue = (
                sales_df.assign(Month=sales_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
                .groupby("Month", as_index=False)["TotalAmount"].sum()
            )
            fig_monthly = px.line(monthly_revenue, x="Month", y="TotalAmount", markers=True, template="plotly_dark", labels={"TotalAmount": "Revenue ($)"})
            fig_monthly.update_traces(line_color=PASTEL_COLORS[0], marker=dict(size=5))
            fig_monthly.update_layout(height=290, hovermode="x unified")
            apply_dark_chart_theme(fig_monthly)
            st.plotly_chart(fig_monthly, use_container_width=True)
            st.caption("Revenue rises sharply in late 2011, indicating strong Q4 seasonality.")

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Revenue by Country</div>', unsafe_allow_html=True)
            country_revenue = transactions.groupby("Country", as_index=False)["TotalAmount"].sum().nlargest(5, "TotalAmount").sort_values("TotalAmount")
            fig_country = px.bar(
                country_revenue,
                x="TotalAmount",
                y="Country",
                orientation="h",
                color="Country",
                color_discrete_sequence=PASTEL_COLORS,
                template="plotly_dark",
                labels={"TotalAmount": "Revenue ($)", "Country": ""},
                hover_data={"Country": True, "TotalAmount": ":,.2f"},
            )
            fig_country.update_layout(showlegend=False, height=290, bargap=0.25)
            apply_dark_chart_theme(fig_country)
            fig_country.update_yaxes(tickfont=dict(color="#FFFFFF", size=10))
            st.plotly_chart(fig_country, use_container_width=True)
            st.caption("The United Kingdom is the dominant market; international markets offer growth potential.")

    # Row 2 Charts
    col_left, col_right = st.columns([3, 2], gap="small")
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
            top_products = sales_df.groupby("Description", as_index=False)["TotalAmount"].sum().nlargest(10, "TotalAmount").sort_values("TotalAmount")
            top_products = top_products.copy()
            top_products["DisplayDescription"] = top_products["Description"].str.slice(0, 28).where(
                top_products["Description"].str.len() <= 28,
                top_products["Description"].str.slice(0, 25) + "..."
            )
            fig_products = px.bar(
                top_products,
                x="TotalAmount",
                y="DisplayDescription",
                orientation="h",
                template="plotly_dark",
                color_discrete_sequence=[PASTEL_COLORS[1]],
                labels={"TotalAmount": "Revenue ($)", "DisplayDescription": ""},
                hover_data={"Description": True, "DisplayDescription": False, "TotalAmount": ":,.2f"},
            )
            fig_products.update_layout(height=310, margin=dict(l=10, r=10, t=20, b=10))
            apply_dark_chart_theme(fig_products)
            fig_products.update_yaxes(title_text="", tickfont=dict(size=10))
            st.plotly_chart(fig_products, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Revenue by Customer Segment</div>', unsafe_allow_html=True)
            segment_summary = df.groupby("Segment_Name", as_index=False).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"), Avg_Frequency=("Frequency", "mean")).sort_values("Revenue", ascending=False)
            segment_plot = segment_summary.copy()
            segment_plot["DisplaySegment"] = segment_plot["Segment_Name"].replace({
                "Champions / High-Value": "Champions",
                "Occasional / Mid-Value": "Occasional",
                "Lapsed / Low-Value": "Lapsed",
            })
            fig_segment = px.bar(
                segment_plot,
                x="DisplaySegment",
                y="Revenue",
                color="Segment_Name",
                color_discrete_sequence=PASTEL_COLORS,
                template="plotly_dark",
                labels={"Revenue": "Revenue ($)", "DisplaySegment": "", "Segment_Name": "Segment"},
                hover_data={"Segment_Name": True, "DisplaySegment": False, "Revenue": ":,.2f"},
            )
            fig_segment.update_layout(showlegend=False, height=220, bargap=0.25)
            apply_dark_chart_theme(fig_segment)
            fig_segment.update_xaxes(title_text="")
            st.plotly_chart(fig_segment, use_container_width=True)
            
            segment_table = (
                segment_summary.style
                .format({"Revenue": "${:,.2f}", "Avg_Frequency": "{:.2f}"})
                .set_properties(**{"background-color": "#151B34", "color": "#F8FAFC"})
                .set_table_styles(
                    [{"selector": "th", "props": [("background-color", "#232D52"), ("color", "#FFFFFF"), ("font-weight", "600")]}]
                )
            )
            st.dataframe(segment_table, hide_index=True, use_container_width=True, height=135)

    # Row 3 Charts
    col_left, col_right = st.columns([3, 2], gap="small")
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Customer Recency / Retention</div>', unsafe_allow_html=True)
            recency_order = ["Active (0-30d)", "Warm (31-90d)", "Cold (91-180d)", "At Risk (181-365d)", "Lapsed (365d+)"]
            recency_summary = df["Recency_Segment"].value_counts().reindex(recency_order, fill_value=0).rename_axis("Recency Segment").reset_index(name="Customers")
            fig_recency = px.bar(recency_summary, x="Recency Segment", y="Customers", color="Recency Segment", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark")
            fig_recency.update_layout(showlegend=False, height=270, bargap=0.25)
            apply_dark_chart_theme(fig_recency)
            fig_recency.update_xaxes(title_text="")
            fig_recency.update_yaxes(title_text="")
            st.plotly_chart(fig_recency, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Guest vs. Registered Orders</div>', unsafe_allow_html=True)
            order_type = pd.DataFrame({"Customer Type": ["Registered", "Guest"], "Orders": [registered_orders, guest_orders]})
            fig_guest = px.pie(order_type, names="Customer Type", values="Orders", hole=0.55, color_discrete_sequence=[PASTEL_COLORS[0], PASTEL_COLORS[4]])
            fig_guest.update_layout(height=250, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
            apply_dark_chart_theme(fig_guest)
            st.plotly_chart(fig_guest, use_container_width=True)
            st.caption("Converting guest purchasers into registered customers supports stronger repeat-purchase analysis.")

with tab2:
    st.subheader("Real-Time Customer Segment Predictor")

    st.markdown(
        """
    <div style="background-color: #151B34; border-left: 4px solid #818CF8; padding: 13px 15px; border-radius: 9px; margin-bottom: 12px;">
        <h4 style="margin: 0 0 8px 0; color: #C4B5FD;">What is this tool and why are we using it?</h4>
        <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0;">
            Imagine a new customer visits your store. Instead of guessing how valuable they are, this tool uses a
            <b>Machine Learning Model</b> to instantly organize them into a group based on 3 simple metrics:
            <br>1. <b>Recency:</b> How many days ago was their last purchase?
            <br>2. <b>Frequency:</b> How many total times have they bought from us?
            <br>3. <b>Monetary:</b> How much total money have they spent?
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("---")

    with st.form("rfm_predict_form"):
        st.markdown("#### Step 1: Enter Customer Activity Data")

        cp1, cp2, cp3 = st.columns(3)

        with cp1:
            input_recency = st.number_input(
                "Recency (Days since last order)",
                min_value=0,
                value=10,
                help="How recently did they buy? Lower numbers mean active buyers.",
            )
        with cp2:
            input_frequency = st.number_input(
                "Frequency (Total orders)",
                min_value=1,
                value=5,
                help="How many times have they checked out?",
            )
        with cp3:
            input_monetary = st.number_input(
                "Monetary Value (Total spent in $)",
                min_value=0.0,
                value=250.0,
                help="Total revenue this customer generated.",
            )

        submit_button = st.form_submit_button("Classify This Customer")

    if submit_button:
        st.write("---")
        st.markdown("#### Step 2: Prediction Results & Action Plan")

        if scaler is not None and kmeans is not None:
            freq_log = np.log1p(input_frequency)
            monetary_log = np.log1p(input_monetary)

            user_input_features = np.array([[freq_log, monetary_log]])
            user_input_scaled = scaler.transform(user_input_features)

            predicted_cluster = kmeans.predict(user_input_scaled)[0]

            segment_info = {
                0: {
                    "name": "Lapsed / Low-Value Customer",
                    "color": "#2A1528",
                    "border": "#F87171",
                    "meaning": "This customer hasn't bought in a long time and has spent very little.",
                    "action": "Send an automated 'We Miss You!' discount code or a short feedback survey.",
                },
                1: {
                    "name": "Champions / High-Value VIP",
                    "color": "#132A3A",
                    "border": "#38BDF8",
                    "meaning": "Your best customers! They buy frequently and spend big amounts.",
                    "action": "Provide early access to new product drops, VIP rewards, and dedicated support.",
                },
                2: {
                    "name": "Occasional / Mid-Value Customer",
                    "color": "#251C3D",
                    "border": "#A78BFA",
                    "meaning": "Average customers. They buy from time to time, but haven't committed to becoming regulars.",
                    "action": "Recommend complementary products or offer a small discount on their next order.",
                },
            }

            info = segment_info.get(
                predicted_cluster,
                {
                    "name": f"Cluster {predicted_cluster}",
                    "color": "#151B34",
                    "border": "#CBD5E0",
                    "meaning": "Assigned to cluster based on statistical similarity.",
                    "action": "Analyze standard marketing response for this group.",
                },
            )

            st.markdown(
                f"""
            <div style="background-color: {info['color']}; border: 2px solid {info['border']}; padding: 16px 18px; border-radius: 11px;">
                <h3 style="margin-top:0;">Result: {info['name']} (Cluster {predicted_cluster})</h3>
                <p style="font-size: 1.05rem; color: #E2E8F0;"><b>What this means:</b> {info['meaning']}</p>
                <hr style="border: 0.5px solid {info['border']};">
                <p style="font-size: 1.05rem; color: #E2E8F0;"><b>Recommended Business Action:</b> {info['action']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "Model files missing. Make sure `scaler.joblib` and `kmeans_model.joblib` are saved under `notebooks/models/`."
            )