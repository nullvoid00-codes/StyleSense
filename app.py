import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64
import altair as alt

from recommendation import (
    RECOMMENDATION_READY,
    RECOMMENDATION_ERROR,
    similarity_load_error,
    customer_ids,
    recommend_products,
    get_customer_profile,
)

# ---------------------------------------------------------
# PAGE CONFIG + PATHS
# ---------------------------------------------------------
st.set_page_config(
    page_title="StyleSense | Fashion Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Locate the final Colab artifacts folder beside app.py.
if (BASE_DIR / "StyleSense_Artifacts").is_dir():
    ARTIFACTS_DIR = BASE_DIR / "StyleSense_Artifacts"
elif (BASE_DIR / "Artifacts").is_dir():
    ARTIFACTS_DIR = BASE_DIR / "Artifacts"
elif (BASE_DIR / "artifacts").is_dir():
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
else:
    ARTIFACTS_DIR = BASE_DIR / "StyleSense_Artifacts"

# ---------------------------------------------------------
# DESIGN SYSTEM — WARM BEIGE, ESPRESSO, EDITORIAL STYLE
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    :root {
        --cream: #F7F3ED;
        --paper: #FFFCF8;
        --sand: #E9DED0;
        --taupe: #B7A18B;
        --brown: #4A3328;
        --espresso: #30221C;
        --muted: #82756B;
        --line: #E8DED3;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--cream);
        color: var(--espresso);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        color: var(--espresso) !important;
        letter-spacing: -0.02em;
    }

    p, label, .stMarkdown, .stCaption {
        color: var(--espresso);
    }

    [data-testid="stSidebar"] {
        background: #EFE6DC;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p {
        color: var(--brown) !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: var(--brown);
        color: #FFFDF9;
        border: 1px solid var(--brown);
        border-radius: 7px;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: var(--espresso);
        border-color: var(--espresso);
        color: white;
    }

    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 18px;
    }

    [data-testid="stTabs"] button {
        color: var(--brown);
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--paper);
        border-radius: 7px;
        border-color: var(--line);
    }

    .hero {
        background:
          linear-gradient(90deg, rgba(36,25,20,.76), rgba(36,25,20,.28)),
          url("https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1800&q=85");
        background-size: cover;
        background-position: center 42%;
        border-radius: 14px;
        padding: 58px 54px;
        min-height: 340px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 28px;
    }

    .hero-kicker {
        color: #E9D9C8;
        font-size: .76rem;
        letter-spacing: .22em;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: #FFFCF8 !important;
        font-size: clamp(2.8rem, 6vw, 5rem);
        line-height: 1.02;
        margin: 0 0 12px 0;
    }

    .hero-copy {
        color: #F5EDE4;
        font-size: 1.05rem;
        max-width: 610px;
        line-height: 1.7;
    }

    .eyebrow {
        color: #947A65;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
    }

    .section-intro {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 22px;
    }

    /* Equal height for all four experience cards */
.feature-card {
    height: 260px !important;
    min-height: 260px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
}
    .feature-number {
        color: #A68B73;
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
    }
    /* Circular profile photo in sidebar */
[data-testid="stSidebar"] [data-testid="stImage"] img {
    width: 72px !important;
    height: 72px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
}

/* Sidebar navigation styled like text links */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #5c493e !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.45rem 0 !important;
    font-size: 16px !important;
}
.stButton > button {
    color: #F8EDE0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: #9b765b !important;
    text-decoration: underline !important;
    background: transparent !important;
}
    .image-caption {
        color: var(--muted);
        font-size: .88rem;
        margin-top: -7px;
    }

    .footer {
        border-top: 1px solid var(--line);
        padding-top: 18px;
        margin-top: 44px;
        color: var(--muted);
        font-size: .82rem;
        text-align: center;
    }

    /* TABLE AND INPUT TEXT VISIBILITY */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        color-scheme: light !important;
        background-color: #FFFCF8 !important;
    }

    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {
        color: #30221C !important;
    }

    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] th {
        color: #4A3328 !important;
        font-weight: 700 !important;
    }

    .stNumberInput input,
    .stTextInput input {
        color: #4A3328 !important;
        -webkit-text-fill-color: #4A3328 !important;
        background-color: #FFFCF8 !important;
        font-weight: 600 !important;
    }

    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label {
        color: #4A3328 !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: #4A3328 !important;
        -webkit-text-fill-color: #4A3328 !important;
    }

    [data-testid="stMetricValue"] {
        color: #4A3328 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #6B5749 !important;
    }

    .stButton button,
    .stFormSubmitButton button {
        color: #FFFFFF !important;
    }

    /* CUSTOMER PROFILE DROPDOWN */

    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background-color: #FFFCF8 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #4A3328 !important;
        -webkit-text-fill-color: #4A3328 !important;
        opacity: 1 !important;
    }

    [data-testid="stSelectbox"] input {
        color: #4A3328 !important;
        -webkit-text-fill-color: #4A3328 !important;
        opacity: 1 !important;
    }

    [data-testid="stSelectbox"] svg {
        fill: #4A3328 !important;
        color: #4A3328 !important;
    }

    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li {
        color: #4A3328 !important;
        background-color: #FFFCF8 !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: #E9DED0 !important;
    }

    /* SIDEBAR */

    [data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: #EFE6DC !important;
        border-right: 1px solid #E1D4C6;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stImage"] img {
        width: 72px !important;
        height: 72px !important;
        object-fit: cover !important;
        border-radius: 50% !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #4A3328 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 7px 9px !important;
        border-radius: 8px !important;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #E4D5C5 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] p {
        color: #4A3328 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #DCCBBB !important;
    }

    [data-testid="stSidebar"] .stCaption {
        color: #82756B !important;
    }
    /* STYLE SENSE - WARM BEIGE TABLE THEME */

/* Table background */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #F8F1E8 !important;
    border: 1px solid #DFD0BF !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Table cells */
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stTable"] th,
[data-testid="stTable"] td {
    background-color: #F8F1E8 !important;
    color: #4B352C !important;
    border-color: #DFD0BF !important;
}

/* Column headers */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stTable"] th {
    background-color: #B79B82 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Hover effect */
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
    background-color: #EFE4D7 !important;
}
/* StyleSense themed information box */
[data-testid="stAlert"] {
    background-color: #F1E5D8 !important;
    border: 1px solid #DFD0BF !important;
    border-left: 5px solid #B79B82 !important;
    border-radius: 10px !important;
}

/* Information box text */
[data-testid="stAlert"] p {
    color: #4B352C !important;
}

/* Information box icon */
[data-testid="stAlert"] svg {
    color: #8B6048 !important;
    fill: #8B6048 !important;
}
div.st-key-generate_recommendations button,
div.st-key-generate_recommendations button p {
    color: #F8EDE0 !important;
}
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p {
    color: #F8EDE0 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------
def require_artifacts():
    if not ARTIFACTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not find the Artifacts folder beside app.py. "
            f"Expected: {ARTIFACTS_DIR}"
        )


def load_pickle(filename):
    require_artifacts()
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return joblib.load(path)


def load_comparison(filename):
    require_artifacts()
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_prediction_models():
    # The final StyleSense project keeps only the Low-Sales Risk model.
    sales_model = load_pickle("sales_risk_model.pkl")
    sales_columns = load_pickle("sales_risk_columns.pkl")
    return sales_model, sales_columns


@st.cache_data(show_spinner=False)
def load_catalog_options():
    catalog = load_pickle("stylesense_catalog.pkl")
    if isinstance(catalog, pd.DataFrame):
        return catalog
    return pd.DataFrame(catalog)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

profile_path = BASE_DIR / "assets" / "sneha.jpeg"
sidebar_bg_path = BASE_DIR / "assets" / "sidebar_bg.png"

if sidebar_bg_path.exists():
    sidebar_bg = base64.b64encode(
        sidebar_bg_path.read_bytes()
    ).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background-image:
                linear-gradient(
                    rgba(245, 237, 226, 0.78),
                    rgba(245, 237, 226, 0.78)
                ),
                url("data:image/png;base64,{sidebar_bg}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if "stylesense_current_page" not in st.session_state:
    st.session_state.stylesense_current_page = "Home"

with st.sidebar:

    profile_col, name_col = st.columns(
    [0.8, 2.7],
    vertical_alignment="center",
    gap="small"
)

    # Profile header
    profile_html = f"""
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        padding:12px 0 22px 0;
    ">
        <img
            src="data:image/jpeg;base64,{__import__('base64').b64encode(profile_path.read_bytes()).decode()}"
            style="
                width:72px;
                height:72px;
                border-radius:50%;
                object-fit:cover;
                flex-shrink:0;
            "
        />
        <div>
            <div style="
                font-family:Georgia,serif;
                font-size:23px;
                font-weight:bold;
                color:#4b352c;
                white-space:nowrap;
            ">StyleSense</div>
            <div style="
                font-size:11px;
                letter-spacing:2px;
                color:#a4866d;
                margin-top:5px;
            ">BY SNEHA PRASAD</div>
        </div>
    </div>
    """

    st.markdown(profile_html, unsafe_allow_html=True)


    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#a4866d;
            font-size:12px;
            font-weight:bold;
            letter-spacing:3px;
            margin-bottom:12px;
        ">EXPLORE</div>
        """,
        unsafe_allow_html=True
    )

    navigation_items = [
        "Home",
        "Fashion Recommendations",
        "ML Predictions",
        "Model Performance",
    ]

    for item in navigation_items:
        if st.button(
            item,
            key=f"navigation_{item.replace(' ', '_')}",
            use_container_width=True
        ):
            st.session_state.stylesense_current_page = item

    page = st.session_state.stylesense_current_page

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#927f70;
            font-size:14px;
            line-height:1.9;
        ">
            A thoughtful blend of fashion discovery and machine learning.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

if page == "Home":

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">
            AI-POWERED FASHION INTELLIGENCE
          </div>
          <h1>Style, with<br>insight.</h1>
          <div class="hero-copy">
            Discover relevant fashion recommendations and explore predictive
            insights from the H&amp;M fashion transaction dataset.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eyebrow">THE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.header("Fashion discovery meets data intelligence")

    st.markdown(
        '<p class="section-intro">One elegant workspace for personalised recommendations, '
        'purchase prediction and product demand insights.</p>',
        unsafe_allow_html=True,
    )

    # Dataset highlights
    m1, m2, m3 = st.columns(3)

    m1.metric("Dataset", "H&M Fashion")
    m2.metric("Transactions", "31.8M")
    m3.metric("Articles", "105,542")

    # Explore sections
    st.markdown("### Explore StyleSense")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.image(
            "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=1000&q=85",
            use_container_width=True,
        )

        st.markdown("#### Fashion Recommendations")

        st.write(
            "Explore customer preferences and generate product suggestions "
            "using similarity, purchase patterns and popularity."
        )

        st.markdown(
            '<div class="image-caption">PERSONALISED DISCOVERY</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.image(
            str(BASE_DIR / "assets" / "predictive_fashion.png"),
            use_container_width=True
        )

        st.markdown("#### Predictive Fashion Analytics")

        st.write(
            "Predict a customer's next purchase category and identify products "
            "that may be at relative risk of low sales."
        )

        st.markdown(
            '<div class="image-caption">PREDICTION & PERFORMANCE</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Built around four simple experiences")

    f1, f2, f3, f4 = st.columns(4)

    for col, number, heading, desc in [
        (f1, "01", "Discover", "Customer-based product recommendations."),
        (f2, "02", "Predict", "Likely next purchase category."),
        (f3, "03", "Assess", "Relative low-sales risk classification."),
        (f4, "04", "Compare", "Saved model evaluation metrics."),
    ]:
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-number">{number}</div>'
                f'<h4>{heading}</h4>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.info(
        "Low-sales risk represents relative demand risk based on the model's "
        "training labels. It does not confirm inventory levels or unsold stock."
    )

# ---------------------------------------------------------
# FASHION RECOMMENDATIONS
# ---------------------------------------------------------

elif page == "Fashion Recommendations":

    st.markdown(
        '<div class="eyebrow">PERSONALISED DISCOVERY</div>',
        unsafe_allow_html=True
    )

    st.title("Fashion Recommendations")

    st.markdown(
        '<p class="section-intro">Choose a customer profile to explore their shopping '
        'preferences and discover relevant products.</p>',
        unsafe_allow_html=True,
    )

    if not RECOMMENDATION_READY:

        st.error("Recommendation artifacts could not be loaded.")

        st.code(str(RECOMMENDATION_ERROR))

        st.info(
            "Check that the Artifacts folder is beside app.py and contains "
            "interaction_matrix.npz, customer_ids.pkl, product_ids.pkl, "
            "product_popularity.pkl and product_details.pkl."
        )

    else:

        if similarity_load_error:
            st.warning(
                "The similarity model could not be loaded. Recommendations may "
                "use the popularity fallback."
            )
            st.caption(str(similarity_load_error))

        visible_customers = customer_ids[:500]

        customer_labels = {
            cid: f"Customer Profile {idx + 1:03d}"
            for idx, cid in enumerate(visible_customers)
        }

        customer_id = st.selectbox(
            "Select a customer profile",
            visible_customers,
            format_func=lambda x: customer_labels[x],
            key="recommendation_customer_select"
        )

        profile = get_customer_profile(customer_id)

        if profile:

            a, b = st.columns(2)

            a.metric(
                "Unique products purchased",
                profile["total_products"]
            )

            top_colour = (
                profile["colours"].index[0]
                if not profile["colours"].empty
                else "Not available"
            )

            b.metric("Top colour", top_colour)

            left, right = st.columns(2)

            with left:

                st.markdown("**Favourite product types**")

                if not profile["categories"].empty:

                    category_table = profile["categories"].rename(
                        "Purchase count"
                    ).to_frame()

                    category_styled = category_table.style.set_properties(
                        **{
                            "background-color": "#F8F1E8",
                            "color": "#4B352C",
                            "border-color": "#DFD0BF"
                        }
                    ).set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#B79B82"),
                                ("color", "#FFFFFF"),
                                ("font-weight", "bold"),
                                ("border-color", "#DFD0BF")
                            ]
                        }
                    ])

                    st.table(category_styled)

                else:
                    st.write("No category history available.")

            with right:

                st.markdown("**Favourite colours**")

                if not profile["colours"].empty:

                    colour_table = profile["colours"].rename(
                        "Purchase count"
                    ).to_frame()

                    colour_styled = colour_table.style.set_properties(
                        **{
                            "background-color": "#F8F1E8",
                            "color": "#4B352C",
                            "border-color": "#DFD0BF"
                        }
                    ).set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#B79B82"),
                                ("color", "#FFFFFF"),
                                ("font-weight", "bold"),
                                ("border-color", "#DFD0BF")
                            ]
                        }
                    ])

                    st.table(colour_styled)

                else:
                    st.write("No colour history available.")

        if st.button(
            "Generate recommendations",
            key="generate_recommendations"
        ):

            with st.spinner("Finding relevant products..."):

                try:

                    recs = recommend_products(
                        customer_id,
                        top_n=8
                    )

                    st.session_state["style_recs"] = recs
                    st.session_state["style_recs_customer"] = customer_id

                except Exception as exc:

                    st.error(
                        f"Could not generate recommendations: {exc}"
                    )

        if (
            st.session_state.get("style_recs_customer") == customer_id
            and "style_recs" in st.session_state
        ):

            recs = st.session_state["style_recs"]

            st.subheader("Recommended for this profile")

            if recs.empty:

                st.warning(
                    "No recommendations were returned for this customer."
                )

            else:

                display_cols = [
                    c for c in [
                        "article_id",
                        "prod_name",
                        "product_type_name",
                        "product_group_name",
                        "colour_group_name",
                        "hybrid_score",
                        "recommendation_method",
                    ]
                    if c in recs.columns
                ]

                recs_styled = recs[display_cols].style.set_properties(
                    **{
                        "background-color": "#F8F1E8",
                        "color": "#4B352C",
                        "border-color": "#DFD0BF",
                    }
                ).set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#B79B82"),
                            ("color", "#FFFFFF"),
                            ("font-weight", "bold"),
                            ("border-color", "#DFD0BF"),
                            ("text-align", "left"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("padding", "10px"),
                            ("border-color", "#DFD0BF"),
                        ],
                    },
                    {
                        "selector": "tbody tr:nth-child(even)",
                        "props": [
                            ("background-color", "#EFE4D7"),
                        ],
                    },
                ])

                st.table(recs_styled)



# ---------------------------------------------------------
# ML PREDICTIONS
# ---------------------------------------------------------

elif page == "ML Predictions":

    st.markdown(
        '<div class="eyebrow">PREDICTIVE ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.title("Low-Sales Risk Prediction")

    st.markdown(
        '<p class="section-intro">Estimate whether a product may fall into the '
        'low-sales group based on its recent sales history and product attributes.</p>',
        unsafe_allow_html=True,
    )

    try:
        sales_model, sales_columns = load_prediction_models()
        models_loaded = True

    except Exception as exc:
        models_loaded = False
        st.error("Could not load the Low-Sales Risk model.")
        st.code(str(exc))
        st.warning(
            "Check that StyleSense_Artifacts contains sales_risk_model.pkl "
            "and sales_risk_columns.pkl."
        )

    if models_loaded:

        st.subheader("Assess product low-sales risk")
        st.caption(
            "Enter the product's historical sales and catalogue attributes."
        )

        catalog = None
        try:
            catalog = load_catalog_options()
        except Exception:
            catalog = None

        if (
            catalog is not None
            and not catalog.empty
            and {
                "product_group_name",
                "product_type_name",
                "colour_group_name"
            }.issubset(catalog.columns)
        ):
            groups = sorted(catalog["product_group_name"].dropna().unique())

            selected_group = st.selectbox(
                "Product group",
                groups,
                key="sales_product_group"
            )

            group_df = catalog[
                catalog["product_group_name"] == selected_group
            ]
            types = sorted(group_df["product_type_name"].dropna().unique())

            selected_type = st.selectbox(
                "Product type",
                types,
                key="sales_product_type"
            )

            type_df = group_df[
                group_df["product_type_name"] == selected_type
            ]
            colours = sorted(type_df["colour_group_name"].dropna().unique())

            selected_colour = st.selectbox(
                "Colour group",
                colours,
                key="sales_colour_group"
            )

        else:
            st.warning(
                "Product catalogue options could not be loaded. "
                "Enter the product attributes manually."
            )
            selected_group = st.text_input(
                "Product group", "Garment Upper body"
            )
            selected_type = st.text_input(
                "Product type", "T-shirt"
            )
            selected_colour = st.text_input(
                "Colour group", "Black"
            )

        with st.form("sales_risk_form"):
            s1, s2 = st.columns(2)

            with s1:
                sales30 = st.number_input(
                    "Sales in previous 30 days",
                    min_value=0,
                    max_value=100000,
                    value=20
                )
                sales60 = st.number_input(
                    "Sales in previous 60 days",
                    min_value=0,
                    max_value=200000,
                    value=45
                )

            with s2:
                sales90 = st.number_input(
                    "Sales in previous 90 days",
                    min_value=0,
                    max_value=300000,
                    value=70
                )
                avg_price90 = st.number_input(
                    "Average price in previous 90 days",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.025,
                    step=0.001,
                    format="%.4f"
                )

            submitted_sales = st.form_submit_button(
                "Predict sales risk",
                type="primary"
            )

        if submitted_sales:
            # Match the exact feature names and order used during Colab training.

            input_values = {
                "sales_30d": sales30,
                "sales_60d": sales60,
                "sales_90d": sales90,
                "average_price": avg_price90,
                "product_group_name": selected_group,
                "product_type_name": selected_type,
                "colour_group_name": selected_colour
            }

            row = pd.DataFrame([input_values])

            row = pd.get_dummies(
                row,
                columns=[
                    "product_group_name",
                    "product_type_name",
                    "colour_group_name"
                ],
                dtype=int
            )

            row = row.reindex(
                columns=sales_columns,
                fill_value=0
            )


            try:
                prediction = sales_model.predict(row)[0]
                predicted_class = int(prediction)

                label = (
                    "Low-sales risk"
                    if predicted_class == 1
                    else "Normal sales"
                )

                risk_probability = None
                if hasattr(sales_model, "predict_proba"):
                    probabilities = sales_model.predict_proba(row)[0]
                    model_classes = list(sales_model.classes_)
                    if 1 in model_classes:
                        risk_probability = float(
                            probabilities[model_classes.index(1)]
                        )

                if predicted_class == 1:
                    st.warning(f"Prediction: **{label}**")
                else:
                    st.success(f"Prediction: **{label}**")

                if risk_probability is not None:
                    st.metric(
                        "Estimated low-sales risk probability",
                        f"{risk_probability:.1%}"
                    )

                st.caption(
                    "This is a model estimate of relative low-sales risk "
                    "based on the training labels. It does not confirm "
                    "that an item is out of stock or unsold."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

elif page == "Model Performance":

    st.markdown(
        '<div class="eyebrow">EVALUATION</div>',
        unsafe_allow_html=True
    )

    st.title("Model Performance")

    st.markdown(
        '<p class="section-intro">A comparison of the evaluation results saved '
        'from the Colab training run.</p>',
        unsafe_allow_html=True,
    )

    # Load saved comparison results
    next_results = load_comparison(
        "next_purchase_model_comparison.csv"
    )

    sales_results = load_comparison(
        "sales_risk_model_comparison.csv"
    )


    # ---------------------------------------------
    # LOW-SALES RISK MODEL COMPARISON
    # ---------------------------------------------

    st.subheader("Low-Sales Risk Prediction")

    if sales_results.empty:

        st.warning(
            "Low-sales comparison CSV not found in Artifacts."
        )

    else:

        sales_results_styled = sales_results.style.set_properties(
            **{
                "background-color": "#F8F1E8",
                "color": "#4B352C",
                "border-color": "#DFD0BF",
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#B79B82"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "bold"),
                    ("border-color", "#DFD0BF"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "10px"),
                    ("border-color", "#DFD0BF"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [
                    ("background-color", "#EFE4D7"),
                ],
            },
        ]).hide(axis="index")

        st.table(sales_results_styled)

        sales_chart_cols = [
            col for col in [
                "Accuracy",
                "Precision_low_risk",
                "Recall_low_risk",
                "F1_low_risk",
            ]
            if col in sales_results.columns
        ]

        if "Algorithm" in sales_results.columns and sales_chart_cols:

            st.markdown("#### Algorithm comparison")

            chart_data = sales_results.set_index("Algorithm")[
                sales_chart_cols
            ]

            # StyleSense themed Low-Sales Risk comparison chart
            sales_chart_long = (
                chart_data.reset_index()
                .melt(
                    id_vars="Algorithm",
                    var_name="Metric",
                    value_name="Score"
                )
            )

            sales_metric_colours = {
                "Accuracy": "#8B6048",
                "Precision_low_risk": "#B79B82",
                "Recall_low_risk": "#D8C1A9",
                "F1_low_risk": "#6B4938",
            }

            sales_performance_chart = (
                alt.Chart(sales_chart_long)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X(
                        "Algorithm:N",
                        title="Algorithm",
                        axis=alt.Axis(
                            labelAngle=0,
                            labelColor="#4B352C",
                            titleColor="#4B352C"
                        )
                    ),
                    xOffset="Metric:N",
                    y=alt.Y(
                        "Score:Q",
                        title="Performance score",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(
                            format=".0%",
                            gridColor="#DFD0BF",
                            labelColor="#4B352C",
                            titleColor="#4B352C"
                        )
                    ),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=list(sales_metric_colours.keys()),
                            range=list(sales_metric_colours.values())
                        ),
                        legend=alt.Legend(
                            title="Evaluation metric",
                            labelColor="#4B352C",
                            titleColor="#4B352C"
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("Algorithm:N"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Score:Q", format=".3f")
                    ]
                )
                .properties(
                    height=350,
                    background="#F8F1E8"
                )
                .configure_view(
                    stroke="#DFD0BF"
                )
            )

            st.altair_chart(
                sales_performance_chart,
                use_container_width=True
            )


        else:

            st.caption(
                "The required metric columns were not found "
                "for the low-sales comparison chart."
            )

    # ---------------------------------------------
    # PERFORMANCE EXPLANATION
    # ---------------------------------------------

    st.markdown("---")

    st.info(
        "These are saved held-out test metrics from Colab. "
        "They are not recalculated when the Streamlit app runs."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    '<div class="footer">'
    'STYLE SENSE &nbsp;·&nbsp; H&amp;M FASHION DATASET '
    '&nbsp;·&nbsp; MACHINE LEARNING PROJECT'
    '</div>',
    unsafe_allow_html=True,
)
