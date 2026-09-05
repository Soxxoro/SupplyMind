import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SupplyMind",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------- GLOBAL -------------------- */

    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* -------------------- SIDEBAR -------------------- */

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] hr {
        border-color: #374151;
    }


    /* -------------------- METRICS -------------------- */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }


    /* -------------------- BUTTON -------------------- */

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 800;
    }


    /* -------------------- HEADER -------------------- */

    .brand-title {
        font-size: 42px;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: -1.5px;
        line-height: 1;
    }

    .brand-subtitle {
        margin-top: 8px;
        font-size: 14px;
        color: #64748b;
    }


    /* -------------------- SECTION -------------------- */

    .section-title {
        font-size: 24px;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 16px;
    }


    /* -------------------- CARD -------------------- */

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        font-family: Arial, sans-serif;
    }

    .card-title {
        font-size: 20px;
        font-weight: 850;
        color: #0f172a;
    }

    .card-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 5px;
        line-height: 1.6;
    }


    /* -------------------- RISK CARD -------------------- */

    .risk-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 28px;
        min-height: 300px;
        text-align: center;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        font-family: Arial, sans-serif;
    }

    .risk-label {
        font-size: 12px;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.1px;
    }

    .risk-score {
        font-size: 62px;
        font-weight: 900;
        color: #0f172a;
        line-height: 1;
        margin-top: 20px;
    }

    .risk-level {
        font-size: 18px;
        font-weight: 900;
        margin-top: 9px;
    }

    .risk-description {
        font-size: 12px;
        color: #64748b;
        line-height: 1.6;
        margin-top: 14px;
    }


    /* -------------------- INFO -------------------- */

    .info-label {
        font-size: 10px;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .info-value {
        font-size: 15px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 4px;
    }


    /* -------------------- DRIVER -------------------- */

    .driver-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }

    .driver-title {
        font-size: 13px;
        font-weight: 800;
        color: #334155;
    }

    .driver-text {
        font-size: 12px;
        color: #64748b;
        line-height: 1.5;
        margin-top: 5px;
    }


    /* -------------------- ACTION -------------------- */

    .action-card {
        border-radius: 16px;
        padding: 21px;
        border: 1px solid #e5e7eb;
        font-family: Arial, sans-serif;
    }

    .action-title {
        font-size: 18px;
        font-weight: 850;
        color: #0f172a;
    }

    .action-text {
        font-size: 13px;
        color: #64748b;
        line-height: 1.7;
        margin-top: 8px;
    }


    /* -------------------- FOOTER -------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 11px;
        padding-top: 30px;
        font-family: Arial, sans-serif;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFE PROGRESS FUNCTION
# ============================================================

def safe_progress(value):
    """
    Convert NumPy/Pandas numeric values into a standard
    Python float before passing them to Streamlit.
    """

    value = float(value)

    value = max(
        0.0,
        min(value, 1.0)
    )

    st.progress(value)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "supplymind_xgboost.pkl",
)

if not os.path.exists(MODEL_PATH):

    st.error(
        "Trained model not found. "
        "Please run supplymind.py first."
    )

    st.stop()


model = joblib.load(
    MODEL_PATH
)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="brand-title">
        SupplyMind
    </div>

    <div class="brand-subtitle">
        Supply Chain Risk Intelligence
        &nbsp;•&nbsp;
        Shipment Decision Support
    </div>
    """
)

st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Shipment Configuration"
    )

    st.caption(
        "Enter the current operational conditions "
        "for a shipment."
    )

    st.divider()

    supplier = st.selectbox(
        "Supplier",
        [
            "Supplier_A",
            "Supplier_B",
            "Supplier_C",
            "Supplier_D",
            "Supplier_E",
        ],
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West",
            "Central",
        ],
    )

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rain",
            "Storm",
            "Flood",
            "Heatwave",
        ],
    )

    shipment_mode = st.selectbox(
        "Shipment Mode",
        [
            "Road",
            "Rail",
            "Air",
            "Sea",
        ],
    )

    product_category = st.selectbox(
        "Product Category",
        [
            "Electronics",
            "Pharma",
            "Food",
            "Automotive",
            "Textile",
        ],
    )

    st.divider()

    st.markdown(
        "### Logistics"
    )

    distance_km = st.number_input(
        "Distance (km)",
        min_value=50.0,
        max_value=1800.0,
        value=500.0,
        step=10.0,
    )

    lead_time_days = st.number_input(
        "Lead Time (days)",
        min_value=1.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
    )

    order_volume = st.number_input(
        "Order Volume",
        min_value=20.0,
        max_value=1200.0,
        value=400.0,
        step=10.0,
    )

    inventory_level = st.number_input(
        "Inventory Level",
        min_value=20.0,
        max_value=1500.0,
        value=500.0,
        step=10.0,
    )

    st.markdown(
        "### Risk Conditions"
    )

    supplier_reliability = st.slider(
        "Supplier Reliability",
        min_value=0.20,
        max_value=1.00,
        value=0.78,
        step=0.01,
    )

    route_risk = st.slider(
        "Route Risk",
        min_value=0.01,
        max_value=1.00,
        value=0.35,
        step=0.01,
    )

    traffic_index = st.slider(
        "Traffic Index",
        min_value=0.01,
        max_value=1.00,
        value=0.50,
        step=0.01,
    )

    st.write("")

    predict_button = st.button(
        "Assess Shipment Risk",
        type="primary",
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:

    st.session_state.prediction_done = False


if predict_button:

    st.session_state.prediction_done = True


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.prediction_done:

    st.html(
        """
        <div class="card">

            <div class="section-title">
                Shipment Risk Overview
            </div>

            <div class="card-subtitle">
                Assess potential disruption before the shipment
                moves through the network.
            </div>

        </div>
        """
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    Monitor
                </div>

                <div class="card-subtitle">
                    Review supplier, weather, route,
                    inventory and logistics conditions.
                </div>

            </div>
            """
        )

    with c2:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    Assess
                </div>

                <div class="card-subtitle">
                    Estimate the likelihood of shipment
                    disruption under current conditions.
                </div>

            </div>
            """
        )

    with c3:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    Act
                </div>

                <div class="card-subtitle">
                    Prioritize shipments that may require
                    additional operational attention.
                </div>

            </div>
            """
        )


# ============================================================
# PREDICTION DASHBOARD
# ============================================================

else:

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    inventory_to_order_ratio = (
        inventory_level /
        (order_volume + 1)
    )

    long_distance = int(
        distance_km > 900
    )

    high_traffic = int(
        traffic_index > 0.70
    )

    supplier_risk = (
        1 - supplier_reliability
    )

    long_lead_time = int(
        lead_time_days > 8
    )

    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "supplier": supplier,
                "region": region,
                "weather": weather,
                "shipment_mode": shipment_mode,
                "product_category": product_category,

                "distance_km":
                    distance_km,

                "supplier_reliability":
                    supplier_reliability,

                "lead_time_days":
                    lead_time_days,

                "order_volume":
                    order_volume,

                "inventory_level":
                    inventory_level,

                "route_risk":
                    route_risk,

                "traffic_index":
                    traffic_index,

                "inventory_to_order_ratio":
                    inventory_to_order_ratio,

                "long_distance":
                    long_distance,

                "high_traffic":
                    high_traffic,

                "supplier_risk":
                    supplier_risk,

                "long_lead_time":
                    long_lead_time,
            }
        ]
    )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    probability = float(
        model.predict_proba(
            input_data
        )[0][1]
    )

    prediction = int(
        model.predict(
            input_data
        )[0]
    )

    risk_percentage = float(
        probability * 100
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk_percentage >= 70:

        risk_level = "HIGH"

        risk_color = "#dc2626"

        risk_background = "#fef2f2"

        status = "AT RISK"

        risk_message = (
            "Immediate operational attention "
            "is recommended for this shipment."
        )

        action_text = (
            "Review the supplier, route and delivery "
            "timeline. Consider contingency planning "
            "before dispatch."
        )

    elif risk_percentage >= 40:

        risk_level = "MEDIUM"

        risk_color = "#d97706"

        risk_background = "#fffbeb"

        status = "MONITOR"

        risk_message = (
            "This shipment should be monitored "
            "closely during its journey."
        )

        action_text = (
            "Monitor route conditions and supplier "
            "reliability. Review the delivery timeline "
            "if conditions worsen."
        )

    else:

        risk_level = "LOW"

        risk_color = "#16a34a"

        risk_background = "#f0fdf4"

        status = "STABLE"

        risk_message = (
            "Current conditions indicate relatively "
            "low disruption risk."
        )

        action_text = (
            "Shipment conditions appear stable. "
            "Continue normal operational monitoring."
        )

    # ========================================================
    # PAGE TITLE
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Shipment Risk Overview
        </div>

        <div class="section-subtitle">
            Current risk assessment based on shipment
            and operational conditions.
        </div>
        """
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Disruption Probability",
            f"{risk_percentage:.1f}%"
        )

    with k2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with k3:

        st.metric(
            "Supplier Reliability",
            f"{float(supplier_reliability):.0%}"
        )

    with k4:

        st.metric(
            "Route Risk",
            f"{float(route_risk):.0%}"
        )

    st.write("")

    # ========================================================
    # MAIN THREE PANELS
    # ========================================================

    left, center, right = st.columns(
        [1.0, 1.45, 1.0]
    )

    # ========================================================
    # CURRENT RISK
    # ========================================================

    with left:

        st.html(
            f"""
            <div class="risk-card">

                <div class="risk-label">
                    Current Risk
                </div>

                <div class="risk-score">
                    {risk_percentage:.1f}%
                </div>

                <div
                    class="risk-level"
                    style="color:{risk_color};"
                >
                    ● {risk_level} RISK
                </div>

                <div class="risk-description">
                    {risk_message}
                </div>

            </div>
            """
        )

        st.write("")

        safe_progress(
            probability
        )

    # ========================================================
    # SHIPMENT OVERVIEW
    # ========================================================

    with center:

        st.html(
            f"""
            <div class="card">

                <div class="card-title">
                    Shipment Overview
                </div>

                <div class="card-subtitle">
                    Current shipment configuration
                </div>

                <div style="
                    display:grid;
                    grid-template-columns:1fr 1fr;
                    gap:22px 32px;
                    margin-top:24px;
                ">

                    <div>
                        <div class="info-label">
                            Supplier
                        </div>

                        <div class="info-value">
                            {supplier}
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Shipment Mode
                        </div>

                        <div class="info-value">
                            {shipment_mode}
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Region
                        </div>

                        <div class="info-value">
                            {region}
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Distance
                        </div>

                        <div class="info-value">
                            {float(distance_km):.0f} km
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Weather
                        </div>

                        <div class="info-value">
                            {weather}
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Lead Time
                        </div>

                        <div class="info-value">
                            {float(lead_time_days):.1f} days
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Product
                        </div>

                        <div class="info-value">
                            {product_category}
                        </div>
                    </div>

                    <div>
                        <div class="info-label">
                            Order Volume
                        </div>

                        <div class="info-value">
                            {float(order_volume):.0f}
                        </div>
                    </div>

                </div>

            </div>
            """
        )

    # ========================================================
    # SHIPMENT STATUS
    # ========================================================

    with right:

        st.html(
            f"""
            <div
                style="
                    background:{risk_background};
                    border:1px solid #e5e7eb;
                    border-radius:18px;
                    padding:28px;
                    min-height:300px;
                    font-family:Arial,sans-serif;
                "
            >

                <div style="
                    font-size:12px;
                    font-weight:800;
                    color:#64748b;
                    text-transform:uppercase;
                    letter-spacing:0.8px;
                ">
                    Shipment Status
                </div>

                <div style="
                    font-size:31px;
                    font-weight:900;
                    color:{risk_color};
                    margin-top:24px;
                ">
                    {status}
                </div>

                <div style="
                    color:#64748b;
                    font-size:13px;
                    line-height:1.7;
                    margin-top:13px;
                ">
                    {
                        "Potential disruption detected "
                        "under current conditions."
                        if prediction == 1
                        else
                        "No major disruption signal detected "
                        "under current conditions."
                    }
                </div>

                <div style="
                    margin-top:28px;
                    font-size:11px;
                    font-weight:800;
                    color:#64748b;
                    text-transform:uppercase;
                ">
                    Risk Probability
                </div>

                <div style="
                    font-size:20px;
                    font-weight:850;
                    color:#0f172a;
                    margin-top:5px;
                ">
                    {risk_percentage:.1f}%
                </div>

            </div>
            """
        )

    st.write("")

    # ========================================================
    # RISK DRIVERS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Operational Risk Drivers
        </div>

        <div class="section-subtitle">
            Conditions that may require additional
            operational attention.
        </div>
        """
    )

    drivers = []

    if supplier_reliability < 0.60:

        drivers.append(
            (
                "Supplier Reliability",
                "Supplier reliability is relatively low."
            )
        )

    if route_risk > 0.60:

        drivers.append(
            (
                "Route Risk",
                "The selected route has elevated risk."
            )
        )

    if traffic_index > 0.70:

        drivers.append(
            (
                "Traffic Conditions",
                "Traffic index indicates elevated congestion."
            )
        )

    if lead_time_days > 8:

        drivers.append(
            (
                "Lead Time",
                "Long delivery lead time increases exposure."
            )
        )

    if weather in [
        "Storm",
        "Flood",
        "Heatwave",
    ]:

        drivers.append(
            (
                "Weather",
                f"Adverse weather condition: {weather}."
            )
        )

    if distance_km > 900:

        drivers.append(
            (
                "Shipment Distance",
                "Long-distance movement increases exposure."
            )
        )

    if inventory_to_order_ratio < 1:

        drivers.append(
            (
                "Inventory Coverage",
                "Inventory may provide limited order coverage."
            )
        )

    if not drivers:

        drivers.append(
            (
                "Stable Conditions",
                "No major operational risk indicators detected."
            )
        )

    d1, d2 = st.columns(2)

    for index, (
        title,
        description
    ) in enumerate(drivers):

        target_column = (
            d1
            if index % 2 == 0
            else d2
        )

        with target_column:

            st.html(
                f"""
                <div class="driver-card">

                    <div class="driver-title">
                        {title}
                    </div>

                    <div class="driver-text">
                        {description}
                    </div>

                </div>
                """
            )

    st.write("")

    # ========================================================
    # DECISION SUPPORT
    # ========================================================

    action_left, action_right = st.columns(
        [1.35, 1]
    )

    with action_left:

        st.html(
            f"""
            <div
                class="action-card"
                style="
                    background:{risk_background};
                "
            >

                <div class="action-title">
                    Recommended Action
                </div>

                <div class="action-text">
                    {action_text}
                </div>

            </div>
            """
        )

    with action_right:

        st.html(
            """
            <div class="action-card">

                <div class="action-title">
                    Inventory Coverage
                </div>

                <div class="action-text">
                    Ratio of available inventory to
                    current order volume.
                </div>

            </div>
            """
        )

        st.metric(
            "Coverage Ratio",
            f"{float(inventory_to_order_ratio):.2f}"
        )

    st.write("")

    # ========================================================
    # OPERATIONAL METRICS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Operational Metrics
        </div>

        <div class="section-subtitle">
            Key indicators associated with the current shipment.
        </div>
        """
    )

    o1, o2, o3, o4 = st.columns(4)

    with o1:

        st.metric(
            "Supplier Reliability",
            f"{float(supplier_reliability):.0%}"
        )

        safe_progress(
            supplier_reliability
        )

    with o2:

        st.metric(
            "Route Risk",
            f"{float(route_risk):.0%}"
        )

        safe_progress(
            route_risk
        )

    with o3:

        st.metric(
            "Traffic Index",
            f"{float(traffic_index):.0%}"
        )

        safe_progress(
            traffic_index
        )

    with o4:

        st.metric(
            "Lead Time",
            f"{float(lead_time_days):.1f} days"
        )

        safe_progress(
            lead_time_days / 20.0
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        SupplyMind · Supply Chain Risk Intelligence
        <br>
        Shipment decision support platform
    </div>
    """
)