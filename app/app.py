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

# 2. MODERN PASTEL STYLING & CUSTOM CSS
PASTEL_COLORS = [
    "#2563EB",
    "#4338CA",
    "#6D28D9",
    "#0E7490",
    "#7E22CE",
]


def apply_dark_chart_theme(fig):
  """Use a consistent dark background and readable labels for every chart."""
  fig.update_layout(
      paper_bgcolor="#151B34",
      plot_bgcolor="#151B34",
      font=dict(color="#F8FAFC"),
      legend=dict(font=dict(color="#F8FAFC")),
  )
  fig.update_xaxes(gridcolor="#2B3560", linecolor="#64748B", tickfont=dict(color="#F8FAFC"), title_font=dict(color="#F8FAFC"))
  fig.update_yaxes(gridcolor="#2B3560", linecolor="#64748B", tickfont=dict(color="#F8FAFC"), title_font=dict(color="#F8FAFC"))
  return fig

st.markdown(
    """
<style>
    /* Global Background and Font Adjustments */
    .stApp {
        background-color: #0B1020;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], .stAppHeader {
        background-color: #080C18 !important;
    }

    [data-testid="stMainBlockContainer"] {
        background-color: #0B1020;
    }

    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp li, .stApp div[data-testid="stMarkdownContainer"] {
        color: #F8FAFC;
    }
    
    /* Modern Header Banner */
    .header-container {
        background: #111827;
        border: 1px solid #312E81;
        padding: 24px;
        border-radius: 16px;
        color: #E2E8F0;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.35);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .header-container p {
        margin-top: 5px;
        font-size: 1.05rem;
        opacity: 0.85;
    }

    /* Custom Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #151B34;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #2B3560;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F1530;
        border-right: 1px solid #26325B;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #151B34 !important;
        color: #F8FAFC !important;
        border-color: #64748B !important;
    }

    div[data-baseweb="select"] * {
        color: #F8FAFC !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #151B34 !important;
        color: #F8FAFC !important;
    }

    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #312E81 !important;
    }

    button[kind="secondary"], button[kind="primary"] {
        background-color: #4338CA !important;
        color: #FFFFFF !important;
        border: 1px solid #818CF8 !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #4338CA !important;
        color: #FFFFFF !important;
        border: 1px solid #818CF8 !important;
    }

    button[kind="secondary"]:hover, button[kind="primary"]:hover {
        background-color: #5B21B6 !important;
        border-color: #C4B5FD !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #5B21B6 !important;
        border-color: #C4B5FD !important;
    }
    
    /* Form & Container Styling */
    .stForm, div[data-testid="stForm"] {
        background-color: #151B34;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #2B3560;
        box-shadow: 0 2px 8px rgba(0,0,0,0.28);
    }

    div[data-testid="stForm"] input {
        background-color: #0F1530 !important;
        color: #F8FAFC !important;
    }

    button[data-baseweb="tab"] {
        color: #F8FAFC !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="header-container">
        <h1> Customer Analytics & Segmentation</h1>
        <p>Explore customer behavior clusters, analyze key business metrics, and predict real-time customer segments.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# 3. DATA & MODEL LOADING


@st.cache_data
def load_data():
  """Build the same registered-customer RFM dataset used in the notebook."""
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
  st.subheader("Performance Summary")
  total_revenue = sales_df["TotalAmount"].sum()
  total_orders = sales_df["InvoiceNo"].nunique()
  average_order_value = total_revenue / total_orders if total_orders else 0
  total_units = sales_df["Quantity"].sum()
  registered_orders = sales_df.loc[~sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
  guest_orders = sales_df.loc[sales_df["CustomerID"].str.startswith("guest_"), "InvoiceNo"].nunique()
  registered_share = registered_orders / total_orders if total_orders else 0
  guest_share = guest_orders / total_orders if total_orders else 0

  k1, k2, k3, k4, k5 = st.columns(5)
  k1.metric("Total Revenue", f"${total_revenue:,.2f}")
  k2.metric("Total Orders", f"{total_orders:,}")
  k3.metric("Average Order Value", f"${average_order_value:,.2f}")
  k4.metric("Total Units Sold", f"{total_units:,}")
  k5.metric("Registered vs. Guest", f"{registered_share:.1%} / {guest_share:.1%}")
  st.caption("Sales metrics reflect the selected country. Customer segmentation uses registered customers only.")
  st.markdown("---")

  col_left, col_right = st.columns(2)
  with col_left:
    st.subheader("Monthly Revenue Trend")
    monthly_revenue = (
        sales_df.assign(Month=sales_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["TotalAmount"].sum()
    )
    fig_monthly = px.line(monthly_revenue, x="Month", y="TotalAmount", markers=True, template="plotly_dark", labels={"TotalAmount": "Revenue ($)"})
    fig_monthly.update_traces(line_color=PASTEL_COLORS[1])
    fig_monthly.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_monthly)
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.caption("Revenue rises sharply in late 2011, indicating strong Q4 seasonality.")

  with col_right:
    st.subheader("Revenue by Country")
    country_revenue = transactions.groupby("Country", as_index=False)["TotalAmount"].sum().nlargest(5, "TotalAmount").sort_values("TotalAmount")
    fig_country = px.bar(country_revenue, x="TotalAmount", y="Country", orientation="h", color="Country", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark", labels={"TotalAmount": "Revenue ($)"})
    fig_country.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_country)
    st.plotly_chart(fig_country, use_container_width=True)
    st.caption("The United Kingdom is the dominant market; international markets offer growth potential.")

  col_left, col_right = st.columns(2)
  with col_left:
    st.subheader("Top 10 Products by Revenue")
    top_products = sales_df.groupby("Description", as_index=False)["TotalAmount"].sum().nlargest(10, "TotalAmount").sort_values("TotalAmount")
    fig_products = px.bar(top_products, x="TotalAmount", y="Description", orientation="h", template="plotly_dark", color_discrete_sequence=[PASTEL_COLORS[0]], labels={"TotalAmount": "Revenue ($)"})
    fig_products.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_products)
    st.plotly_chart(fig_products, use_container_width=True)

  with col_right:
    st.subheader("Revenue by Customer Segment")
    segment_summary = df.groupby("Segment_Name", as_index=False).agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"), Avg_Frequency=("Frequency", "mean")).sort_values("Revenue", ascending=False)
    fig_segment = px.bar(segment_summary, x="Segment_Name", y="Revenue", color="Segment_Name", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark", labels={"Revenue": "Revenue ($)", "Segment_Name": "Segment"})
    fig_segment.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_segment)
    st.plotly_chart(fig_segment, use_container_width=True)
    segment_table = (
        segment_summary.style
        .format({"Revenue": "${:,.2f}", "Avg_Frequency": "{:.2f}"})
        .set_properties(**{"background-color": "#151B34", "color": "#F8FAFC"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#312E81"),
                        ("color", "#FFFFFF"),
                        ("font-weight", "600"),
                    ],
                }
            ]
        )
    )
    st.dataframe(segment_table, hide_index=True, use_container_width=True)

  col_left, col_right = st.columns(2)
  with col_left:
    st.subheader("Customer Recency / Retention")
    recency_order = ["Active (0-30d)", "Warm (31-90d)", "Cold (91-180d)", "At Risk (181-365d)", "Lapsed (365d+)"]
    recency_summary = df["Recency_Segment"].value_counts().reindex(recency_order, fill_value=0).rename_axis("Recency Segment").reset_index(name="Customers")
    fig_recency = px.bar(recency_summary, x="Recency Segment", y="Customers", color="Recency Segment", color_discrete_sequence=PASTEL_COLORS, template="plotly_dark")
    fig_recency.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_recency)
    st.plotly_chart(fig_recency, use_container_width=True)

  with col_right:
    st.subheader("Guest vs. Registered Orders")
    order_type = pd.DataFrame({"Customer Type": ["Registered", "Guest"], "Orders": [registered_orders, guest_orders]})
    fig_guest = px.pie(order_type, names="Customer Type", values="Orders", hole=0.55, color_discrete_sequence=[PASTEL_COLORS[0], PASTEL_COLORS[4]])
    fig_guest.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    apply_dark_chart_theme(fig_guest)
    st.plotly_chart(fig_guest, use_container_width=True)
    st.caption("Converting guest purchasers into registered customers supports stronger repeat-purchase analysis.")
with tab2:
  # -------------------------------------------------------------------------
  # SECTION 5: REAL-TIME CUSTOMER PREDICTOR (WITH PLAIN-ENGLISH NOTES)
  # -------------------------------------------------------------------------
  st.subheader(" Real-Time Customer Segment Predictor")

  # Explanatory Guide Box for Non-Sales Users
  st.markdown(
      """
    <div style="background-color: #151B34; border-left: 4px solid #818CF8; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 8px 0; color: #C4B5FD;"> What is this tool and why are we using it?</h4>
        <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0;">
            Imagine a new customer visits your store. Instead of guessing how valuable they are, this tool uses your 
            <b>Machine Learning Model</b> to instantly organize them into a group based on 3 simple questions:
            <br>1. <b>Recency:</b> How many days ago was their last purchase?
            <br>2. <b>Frequency:</b> How many total times have they bought from us?
            <br>3. <b>Monetary:</b> How much total money have they spent?
        </p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.write("---")

  # Prediction Input Form
  with st.form("rfm_predict_form"):
    st.markdown("####  Step 1: Enter Customer Activity Data")

    cp1, cp2, cp3 = st.columns(3)

    with cp1:
      input_recency = st.number_input(
          "Recency (Days since last order)",
          min_value=0,
          value=10,
          help=(
              "How recently did they buy? Lower numbers mean they are active"
              " buyers!"
          ),
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

    submit_button = st.form_submit_button(" Classify This Customer")

  # Output & Explanation Section
  if submit_button:
    st.write("---")
    st.markdown("####  Step 2: Prediction Results & Action Plan")

    if scaler is not None and kmeans is not None:
      # Apply Log Transform & Scaling behind the scenes
      freq_log = np.log1p(input_frequency)
      monetary_log = np.log1p(input_monetary)

      user_input_features = np.array([[freq_log, monetary_log]])
      user_input_scaled = scaler.transform(user_input_features)

      # Get Cluster Prediction (0, 1, or 2)
      predicted_cluster = kmeans.predict(user_input_scaled)[0]

      # Human-friendly dictionary mapping clusters to business advice
      segment_info = {
          0: {
              "name": "Lapsed / Low-Value Customer",
              "color": "#2A1528",
              "border": "#F87171",
              "icon": "",
              "meaning": (
                  "This customer hasn't bought in a long time and has spent very"
                  " little."
              ),
              "action": (
                  "Send an automated 'We Miss You!' discount code or a short"
                  " feedback survey to see why they stopped purchasing."
              ),
          },
          1: {
              "name": "Champions / High-Value VIP",
              "color": "#132A3A",
              "border": "#38BDF8",
              "icon": "",
              "meaning": (
                  "Your best customers! They buy frequently, spend big"
                  " amounts, and purchased recently."
              ),
              "action": (
                  "Do NOT spam them with basic discounts. Give them early"
                  " access to new product drops, VIP rewards, and dedicated"
                  " support."
              ),
          },
          2: {
              "name": "Occasional / Mid-Value Customer",
              "color": "#251C3D",
              "border": "#A78BFA",
              "icon": "",
              "meaning": (
                  "Average customers. They buy from time to time, but haven't"
                  " committed to becoming regulars yet."
              ),
              "action": (
                  "Recommend complementary products (upselling) or offer a small"
                  " discount on their next order to build habit."
              ),
          },
      }

      # Retrieve selected segment details
      info = segment_info.get(
          predicted_cluster,
          {
              "name": f"Cluster {predicted_cluster}",
              "color": "#151B34",
              "border": "#CBD5E0",
              "icon": "",
              "meaning": "Assigned to cluster based on statistical similarity.",
              "action": "Analyze standard marketing response for this group.",
          },
      )

      # Display Structured Result Card
      st.markdown(
          f"""
        <div style="background-color: {info['color']}; border: 2px solid {info['border']}; padding: 20px; border-radius: 12px;">
            <h3 style="margin-top:0;">{info['icon']} Result: {info['name']} (Cluster {predicted_cluster})</h3>
            <p style="font-size: 1.05rem; color: #E2E8F0;"><b>What this means:</b> {info['meaning']}</p>
            <hr style="border: 0.5px solid {info['border']};">
            <p style="font-size: 1.05rem; color: #E2E8F0;"><b>Recommended Business Action:</b> {info['action']}</p>
        </div>
        """,
          unsafe_allow_html=True,
      )

    else:
      st.warning(
          " Model files missing. Make sure `scaler.joblib` and"
          " `kmeans_model.joblib` are saved under `notebooks/models/`."
      )
