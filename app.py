import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import shap
import warnings



warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Intelligence Hub",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #C9A84C;
    --gold-light: #E8C97A;
    --teal: #0D4F4F;
    --teal-light: #1A7A7A;
    --cream: #F5F0E8;
    --dark: #0A0F0F;
    --card-bg: #0F1A1A;
    --border: rgba(201,168,76,0.25);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--cream);
}

.stApp { background: linear-gradient(135deg, #0A0F0F 0%, #0D1F1F 50%, #0A1515 100%); }

/* Hide default streamlit elements */
#MainMenu, footer { visibility: hidden; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, rgba(13,79,79,0.6) 0%, rgba(10,15,15,0.9) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '◆';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    color: rgba(201,168,76,0.06);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--gold);
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(245,240,232,0.6);
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--gold); }
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
}
.metric-label {
    font-size: 0.75rem;
    color: rgba(245,240,232,0.5);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* Segment result card */
.segment-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid var(--border);
    text-align: center;
}
.segment-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.segment-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Recommendation box */
.rec-box {
    background: rgba(201,168,76,0.08);
    border-left: 3px solid var(--gold);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: var(--cream);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}

/* Input sliders */
.stSlider label { color: var(--cream) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
    color: var(--dark) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    font-size: 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(201,168,76,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border-radius: 8px;
    border: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(245,240,232,0.5) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    background: rgba(201,168,76,0.1) !important;
}

/* Plotly charts dark bg */
.js-plotly-plot { border-radius: 12px; }

/* Divider */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 2rem 0;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;   /* spacing between tabs */
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px;
    padding: 10px 18px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load('models/clv_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, scaler, explainer

try:
    model, scaler, explainer = load_models()
    models_loaded = True
except:
    models_loaded = False

# ── Data ───────────────────────────────────────────────────────────────────────
SEGMENTS = {
    0: {
        "name": "Lost Customers",
        "emoji": "🔴",
        "color": "#E05555",
        "bg": "rgba(224,85,85,0.08)",
        "border": "rgba(224,85,85,0.3)",
        "badge_bg": "rgba(224,85,85,0.15)",
        "description": "Customers who haven't engaged in a long time with low purchase history.",
        "recommendations": [
            "Launch a win-back email campaign with an exclusive discount",
            "Survey them to understand why they left",
            "Offer a one-time loyalty bonus to re-engage",
            "Consider reducing marketing spend if no response after 2 attempts"
        ],
        "clv_estimate": "$0 – $500",
        "priority": "Low",
        "action": "Re-engage or Deprioritize"
    },
    1: {
        "name": "Medium Value",
        "emoji": "🟡",
        "color": "#C9A84C",
        "bg": "rgba(201,168,76,0.08)",
        "border": "rgba(201,168,76,0.3)",
        "badge_bg": "rgba(201,168,76,0.15)",
        "description": "Active customers with moderate purchase frequency and spend.",
        "recommendations": [
            "Offer loyalty program membership to increase frequency",
            "Upsell complementary products based on purchase history",
            "Send personalized product recommendations monthly",
            "Provide early access to sales to build loyalty"
        ],
        "clv_estimate": "$500 – $5,000",
        "priority": "Medium",
        "action": "Nurture & Upsell"
    },
    2: {
        "name": "High Value",
        "emoji": "🟢",
        "color": "#4CAF82",
        "bg": "rgba(76,175,130,0.08)",
        "border": "rgba(76,175,130,0.3)",
        "badge_bg": "rgba(76,175,130,0.15)",
        "description": "Frequent buyers with significant spend. Core revenue drivers.",
        "recommendations": [
            "Enroll in VIP membership with exclusive perks",
            "Assign a dedicated account manager",
            "Offer premium product previews before public launch",
            "Create a referral program — they'll bring similar customers"
        ],
        "clv_estimate": "$5,000 – $80,000",
        "priority": "High",
        "action": "Retain & Reward"
    },
}

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='rgba(245,240,232,0.7)', size=12),
    margin=dict(l=20, r=20, t=40, b=20)
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.4rem; color: #C9A84C; font-weight: 700;'>
            💎 CLV Hub
        </div>
        <div style='font-size: 0.7rem; color: rgba(245,240,232,0.4); letter-spacing: 2px; text-transform: uppercase; margin-top: 0.2rem;'>
            Intelligence Dashboard
        </div>
    </div>
    <hr style='border-color: rgba(201,168,76,0.2);'>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Customer Profile")
    st.markdown("<div style='font-size:0.8rem; color:rgba(245,240,232,0.4); margin-bottom:1rem;'>Adjust the RFM parameters below</div>", unsafe_allow_html=True)

    recency = st.slider("**📅 Recency (days)**", 1, 700, 100,
                        help="Days since last purchase. Lower = more recent.")

    frequency = st.slider("**🔁 Frequency (orders)**", 1, 250, 10,
                          help="Total number of unique orders placed.")

    monetary = st.number_input("**💲 Monetary Value ($)**", 1.0, 500000.0, 1000.0, step=100.0,
                               help="Total amount spent by the customer.")

    st.markdown("<hr style='border-color: rgba(201,168,76,0.2);'>", unsafe_allow_html=True)

    predict_btn = st.button("⚡ Analyse Customer", use_container_width=True)

    st.markdown("<hr style='border-color: rgba(201,168,76,0.2);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:rgba(245,240,232,0.3); text-align:center; line-height:1.8;'>
        Model: Random Forest + Class Weights<br>
        Features: RFM Analysis<br>
        Dataset: UCI Online Retail II
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Customer Lifetime Value Intelligence</h1>
    <p>Advanced RFM segmentation · SHAP explainability · Business strategy engine</p>
</div>
""", unsafe_allow_html=True)


# ── Default state ──────────────────────────────────────────────────────────────
if not predict_btn:
    # Show overview metrics
    col1, col2, col3 = st.columns(3)
    overview = [
        ("5,878", "Total Customers Analysed"),
        ("3", "Distinct Segments"),
        ("100%", "Model Accuracy"),
        ("1.00", "Macro F1 Score"),
    ]
    for col, (val, label) in zip([col1, col2, col3], overview):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Segment overview cards
    st.markdown("<div class='section-header'>Segment Intelligence Map</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (k, seg) in zip(cols, SEGMENTS.items()):
        with col:
            st.markdown(f"""
            <div style='background:{seg["bg"]}; border:1px solid {seg["border"]}; border-radius:12px; padding:1.2rem; text-align:center;'>
                <div style='font-size:2rem;'>{seg["emoji"]}</div>
                <div style='font-family: Playfair Display, serif; font-size:1rem; color:{seg["color"]}; font-weight:700; margin:0.5rem 0;'>{seg["name"]}</div>
                <div style='font-size:0.7rem; color:rgba(245,240,232,0.5); margin-bottom:0.8rem;'>{seg["description"][:60]}...</div>
                <div style='background:{seg["badge_bg"]}; border-radius:50px; padding:0.2rem 0.8rem; font-size:0.7rem; color:{seg["color"]}; display:inline-block;'>{seg["action"]}</div>
                <div style='margin-top:0.8rem; font-size:0.75rem; color:rgba(245,240,232,0.4);'>Est. CLV: <span style='color:{seg["color"]};'>{seg["clv_estimate"]}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:rgba(245,240,232,0.35); font-size:0.85rem; padding: 1rem;'>
        ← Adjust customer parameters in the sidebar and click <strong style='color:#C9A84C;'>Analyse Customer</strong> to get a prediction
    </div>
    """, unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
else:
    if not models_loaded:
        st.error("⚠️ Models not found. Make sure clv_model.pkl, scaler.pkl, and shap_explainer.pkl are in the models/ folder.")
        st.stop()

    input_data = np.array([[recency, frequency, monetary]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    seg = SEGMENTS[prediction]

    # ── Result card ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='segment-card' style='background:{seg["bg"]}; border-color:{seg["border"]};'>
        <div style='font-size:3rem;'>{seg["emoji"]}</div>
        <div style='font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:{seg["color"]}; margin-bottom:0.3rem;'>Customer Classification</div>
        <div class='segment-title' style='color:{seg["color"]};'>{seg["name"]}</div>
        <div style='background:{seg["badge_bg"]}; display:inline-block; padding:0.3rem 1.2rem; border-radius:50px; font-size:0.75rem; color:{seg["color"]}; margin:0.5rem 0;'>
            {seg["action"]} · Est. CLV: {seg["clv_estimate"]}
        </div>
        <div style='color:rgba(245,240,232,0.6); font-size:0.9rem; max-width:500px; margin:0.8rem auto 0;'>{seg["description"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🧠 SHAP Explainability", "💡 Strategy", "📈 Benchmarks"])

    # ── Tab 1: Analytics ───────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<div class='section-header'>Confidence Scores</div>", unsafe_allow_html=True)
            fig = go.Figure()
            seg_names = [
                ("Diamonds" if SEGMENTS[i]["name"] == "Champions" else SEGMENTS[i]["name"].upper())
                for i in range(3)
            ]
            seg_colors = [SEGMENTS[i]["color"] for i in range(3)]
            colors = [
                seg_colors[i] if i == prediction else 'rgba(255,255,255,0.08)'
                for i in range(3)
            ]

            fig.add_trace(go.Bar(
                x=prediction_proba,
                y=seg_names,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                text=[f"{p * 100:.1f}%" for p in prediction_proba],
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig.update_layout(
                **PLOTLY_THEME,
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    range=[0, 1]
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(
                        size=13,
                        family="DM Sans",
                        color="rgba(245,240,232,0.9)"
                    )
                ),
                height=220,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>RFM Radar Profile</div>", unsafe_allow_html=True)
            # Normalize values for radar
            r_norm = max(0, 1 - recency/700)
            f_norm = min(frequency/250, 1)
            m_norm = min(monetary/500000, 1)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=[r_norm, f_norm, m_norm, r_norm],
                theta=['Recency', 'Frequency', 'Monetary', 'Recency'],
                fill='toself',
                fillcolor=f'rgba({int(seg["color"][1:3],16)},{int(seg["color"][3:5],16)},{int(seg["color"][5:7],16)},0.2)',
                line=dict(color=seg["color"], width=2),
                name='Customer'
            ))
            fig2.update_layout(
                **PLOTLY_THEME,
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='rgba(245,240,232,0.6)')
                ),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # RFM Summary metrics
        st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        summaries = [
            ("📅", f"{recency} days", "Recency", "Lower is better"),
            ("🔁", f"{frequency} orders", "Frequency", "Higher is better"),
            ("💷", f"£{monetary:,.0f}", "Monetary", "Higher is better"),
            ("🎯", f"{prediction_proba[prediction]*100:.0f}%", "Confidence", "Model certainty"),
        ]
        for col, (icon, val, label, hint) in zip([m1, m2, m3], summaries):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:1.5rem;'>{icon}</div>
                    <div class='metric-value' style='font-size:1.5rem;'>{val}</div>
                    <div class='metric-label'>{label}</div>
                    <div style='font-size:0.65rem; color:rgba(245,240,232,0.3); margin-top:0.3rem;'>{hint}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: SHAP ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-header'>Why This Prediction?</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:rgba(245,240,232,0.5); font-size:0.85rem; margin-bottom:1rem;'>SHAP values show how each feature pushed the model towards <strong style='color:{seg['color']};'>{seg['name']}</strong></div>", unsafe_allow_html=True)


        try:
            shap_values = explainer(input_scaled).values
            feature_names = ['Recency', 'Frequency', 'Monetary']
            sv = shap_values[0, :, prediction]

            # Waterfall chart using plotly
            base = float(explainer.expected_value[prediction])
            final = base + sv.sum()

            colors = [seg["color"] if v >= 0 else "#E05555" for v in sv]
            fig3 = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=feature_names + ["Prediction"],
                y=list(sv) + [None],
                base=base,
                connector=dict(line=dict(color="rgba(201,168,76,0.3)", width=1)),
                increasing=dict(marker=dict(color=seg["color"])),
                decreasing=dict(marker=dict(color="#E05555")),
                totals=dict(marker=dict(color="rgba(201,168,76,0.9)"))
            ))
            fig3.update_layout(
                **PLOTLY_THEME,
                height=350,
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(showgrid=False),
                title=dict(text="Feature Contribution Waterfall", font=dict(color='rgba(245,240,232,0.7)', size=13))
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Feature impact table
            st.markdown("<div class='section-header'>Feature Impact Breakdown</div>", unsafe_allow_html=True)
            for fname, fval, sval in zip(feature_names, input_data[0], sv):
                direction = "▲ Increases" if sval >= 0 else "▼ Decreases"
                dir_color = seg["color"] if sval >= 0 else "#E05555"
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:0.8rem 1.2rem; margin:0.4rem 0; display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:rgba(245,240,232,0.8); font-weight:500;'>{fname}</span>
                    <span style='color:rgba(245,240,232,0.4); font-size:0.85rem;'>Value: <strong style='color:rgba(245,240,232,0.7);'>{fval:,.1f}</strong></span>
                    <span style='color:{dir_color}; font-size:0.85rem; font-weight:500;'>{direction} {seg["name"]} probability</span>
                    <span style='color:{dir_color}; font-weight:700;'>{sval:+.4f}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP visualization error: {e}")

    # ── Tab 3: Strategy ────────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"<div class='section-header'>Strategic Playbook for {seg['name']}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Recommended Actions:**")
            for i, rec in enumerate(seg["recommendations"], 1):
                st.markdown(f"<div class='rec-box'>{'🥇' if i==1 else '🔹'} {rec}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background:{seg["bg"]}; border:1.2px solid {seg["border"]}; border-radius:12px; padding:1.5rem; text-align:center; margin-top: 3.06rem;'>
                <div style='font-size:0.9rem; font-weight:800; letter-spacing:2px; color:{seg["color"]}; text-transform:uppercase; margin-bottom:1rem;'>Segment Profile</div>
                <div style='margin:0.8rem 0;'>
                    <div style='font-size:0.7rem; color:rgba(245,240,232,0.4);'>Est. Lifetime Value</div>
                    <div style='font-size:1.2rem; color:{seg["color"]}; font-weight:700; font-family: Playfair Display, serif;'>{seg["clv_estimate"]}</div>
                </div>
                <div style='margin:0.8rem 0;'>
                    <div style='font-size:0.7rem; color:rgba(245,240,232,0.4);'>Business Priority</div>
                    <div style='font-size:1.2rem; color:{seg["color"]}; font-weight:700;'>{seg["priority"]}</div>
                </div>
                <div style='margin:0.8rem 0;'>
                    <div style='font-size:0.7rem; color:rgba(245,240,232,0.4);'>Strategy</div>
                    <div style='font-size:0.9rem; color:{seg["color"]}; font-weight:600;'>{seg["action"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


    def hex_to_rgba(hex_color, alpha=0.3):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    # ── Tab 4: Benchmarks ──────────────────────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-header'>How Does This Customer Compare?</div>", unsafe_allow_html=True)

        benchmarks = {
            "Lost Customers": {"Recency": 462, "Frequency": 2, "Monetary": 749},
            "Medium Value": {"Recency": 67, "Frequency": 7, "Monetary": 2949},
            "High Value": {"Recency": 26, "Frequency": 104, "Monetary": 81356},
        }

        for metric in ["Recency", "Frequency", "Monetary"]:
            user_val = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}[metric]
            seg_avgs = [benchmarks[s][metric] for s in benchmarks]
            seg_names_list = list(benchmarks.keys())
            seg_colors_list = [SEGMENTS[i]["color"] for i in range(3)]

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=seg_names_list,
                y=seg_avgs,
                marker_color=[hex_to_rgba(s, 0.3) for s in seg_colors_list],
                marker_line_color=seg_colors_list,
                marker_line_width=1.5,
                name="Segment Average"
            ))
            fig_b.add_hline(
                y=user_val,
                line_dash="dash",
                line_color="#C9A84C",
                annotation_text=f"  Your customer: {user_val:,.0f}",
                annotation_font_color="#C9A84C"
            )
            fig_b.update_layout(
                **PLOTLY_THEME,
                title=dict(text=f"{metric} Comparison", font=dict(color='rgba(245,240,232,0.7)', size=13)),
                height=220,
                showlegend=False,
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_b, use_container_width=True)