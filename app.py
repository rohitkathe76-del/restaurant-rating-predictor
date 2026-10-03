import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Zomato Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — restaurant theme, warm palette
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d1b1b 50%, #1a1a2e 100%);
        background-attachment: fixed;
    }

    .hero {
        background: linear-gradient(120deg, #d64541 0%, #e67e22 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(214, 69, 65, 0.35);
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    .hero::after {
        content: "🍕 🍔 🍜 🍣 🥗 🍰";
        position: absolute;
        right: -10px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2.5rem;
        opacity: 0.15;
        letter-spacing: 12px;
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        color: white;
        font-size: 2.2rem;
        margin: 0;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b1b 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #ffd166;
        font-family: 'Poppins', sans-serif;
    }

    .stButton>button {
        background: linear-gradient(120deg, #d64541, #e67e22);
        color: white;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 6px 18px rgba(214, 69, 65, 0.4);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(214, 69, 65, 0.55);
    }

    .stMarkdown, p, label, .stSelectbox label, .stNumberInput label, .stRadio label, .stSlider label {
        color: #f0f0f0 !important;
    }

    .footer-note {
        text-align: center;
        color: rgba(255,255,255,0.45);
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    /* Landing page — step cards */
    .step-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .step-card:hover {
        transform: translateY(-4px);
        border-color: rgba(230, 126, 34, 0.5);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(120deg, #d64541, #e67e22);
        color: white;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 0.7rem;
        font-size: 0.95rem;
    }
    .step-emoji { font-size: 2.1rem; margin-bottom: 0.4rem; }
    .step-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #ffd166;
        font-size: 1.02rem;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        color: rgba(255,255,255,0.65);
        font-size: 0.88rem;
        line-height: 1.4;
    }

    /* Cuisine chip strip */
    .chip-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.6rem;
        margin: 0.5rem 0 1.8rem 0;
    }
    .chip {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-size: 0.88rem;
        color: #f0f0f0;
    }

    .section-heading {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #f0f0f0;
        font-size: 1.15rem;
        text-align: center;
        margin: 0.5rem 0 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🍽️ Restaurant Rating Predictor</h1>
    <p>Fill in your restaurant's details to get an instant predicted rating</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    target_encoding_maps = joblib.load('target_encoding_maps.pkl')
    global_mean = joblib.load('global_mean.pkl')
    final_columns = joblib.load('final_columns.pkl')
    dropdown_values = joblib.load('dropdown_values.pkl')
    return model, target_encoding_maps, global_mean, final_columns, dropdown_values

try:
    model, target_encoding_maps, global_mean, final_columns, dropdown_values = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False
    st.error("Model files not found. Run `train_and_save_model.py` first to generate the required .pkl files.")

# ============================================================
# SIDEBAR — INPUTS ONLY
# ============================================================
with st.sidebar:
    st.markdown("## 🧾 Restaurant Details")
    st.markdown("---")

    if artifacts_loaded:
        st.markdown("#### 📍 Location & Identity")
        location = st.selectbox("Location", dropdown_values['location'])
        name = st.selectbox("Restaurant Name", dropdown_values['name'])

        st.markdown("#### 🍴 Food & Service")
        rest_type = st.selectbox("Restaurant Type", dropdown_values['rest_type'])
        cuisines = st.selectbox("Primary Cuisine", dropdown_values['cuisines'])
        listed_in_type = st.selectbox("Listing Category", dropdown_values['listed_in_type'])
        listed_in_city = st.selectbox("City Zone", dropdown_values['listed_in_city'])

        st.markdown("#### ⚙️ Features")
        online_order = st.radio("Online Order Available?", dropdown_values['online_order'], horizontal=True)
        book_table = st.radio("Table Booking Available?", dropdown_values['book_table'], horizontal=True)

        st.markdown("#### 💰 Cost & Popularity")
        cost_for_2 = st.slider("Approx. Cost for Two (₹)", min_value=50, max_value=6000, value=500, step=50)
        votes = st.number_input("Number of Votes", min_value=0, value=50, step=10)

        st.markdown("---")
        predict_clicked = st.button("🔮 Predict Rating", type="primary")
    else:
        predict_clicked = False

# ============================================================
# MAIN AREA — RESULT ONLY
# ============================================================
if artifacts_loaded and predict_clicked:
    input_dict = {
        'online_order': online_order,
        'book_table': book_table,
        'votes': votes,
        'location': location,
        'rest_type': rest_type,
        'cuisines': cuisines,
        'cost_for_2': cost_for_2,
        'listed_in_type': listed_in_type,
        'listed_in_city': listed_in_city,
        'name': name
    }
    input_df = pd.DataFrame([input_dict])

    high_card_cols = ['location', 'cuisines', 'listed_in_city', 'name']
    for col in high_card_cols:
        input_df[col] = input_df[col].map(target_encoding_maps[col]).fillna(global_mean)

    low_card_cols = ['online_order', 'book_table', 'rest_type', 'listed_in_type']
    input_df = pd.get_dummies(input_df, columns=low_card_cols, drop_first=True)
    input_df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in input_df.columns]
    input_df = input_df.reindex(columns=final_columns, fill_value=0)

    prediction = float(np.clip(model.predict(input_df)[0], 1.0, 5.0))
    delta_vs_avg = prediction - global_mean

    # ---- Build plain-language insights from what the user entered ----
    insights = []

    if cost_for_2 <= 300:
        insights.append(("💰", "Budget-Friendly", "Low price point tends to attract high footfall and volume-driven reviews."))
    elif cost_for_2 <= 800:
        insights.append(("💰", "Mid-Range Pricing", "This is the most common price bracket — competition here is high."))
    else:
        insights.append(("💰", "Premium Pricing", "Higher price points usually come with higher service expectations."))

    if votes < 20:
        insights.append(("👥", "New / Low Visibility", "Few votes so far — ratings can swing more until more reviews come in."))
    elif votes < 200:
        insights.append(("👥", "Building Reputation", "A healthy number of votes — rating is becoming more reliable."))
    else:
        insights.append(("👥", "Well-Established", "High vote count — this rating reflects strong, consistent customer feedback."))

    if online_order == "Yes" and book_table == "Yes":
        insights.append(("⚙️", "Full Service", "Offering both online ordering and table booking tends to boost convenience scores."))
    elif online_order == "Yes" or book_table == "Yes":
        insights.append(("⚙️", "Partial Convenience", "Offering just one of online ordering / table booking — the other could help."))
    else:
        insights.append(("⚙️", "Dine-in Focused", "No online ordering or booking — fine for a traditional dine-in experience."))

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix': " / 5", 'font': {'size': 54, 'color': '#ffffff', 'family': 'Poppins'}},
            gauge={
                'axis': {'range': [1, 5], 'tickcolor': 'white', 'tickwidth': 1, 'tickfont': {'color': 'white'}},
                'bar': {'color': "#ffffff", 'thickness': 0.28},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.3)",
                'steps': [
                    {'range': [1, 2.5], 'color': '#e74c3c'},
                    {'range': [2.5, 3.5], 'color': '#f39c12'},
                    {'range': [3.5, 4.2], 'color': '#f1c40f'},
                    {'range': [4.2, 5], 'color': '#2ecc71'}
                ],
                'threshold': {
                    'line': {'color': '#ffd166', 'width': 4},
                    'thickness': 0.8,
                    'value': prediction
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=300,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        if prediction >= 4.2:
            st.success(f"🌟 Excellent! **{name}** is predicted to be a top-tier restaurant in {location} — keep up the great work!")
        elif prediction >= 3.5:
            st.info(f"👍 Great going! **{name}** is predicted to be a solid, well-regarded choice in {location}.")
        elif prediction >= 2.5:
            st.warning(f"🌱 **{name}** is on a promising path in {location} — a few tweaks could take it to the next level!")
        else:
            st.info(f"💪 Every great restaurant starts somewhere! **{name}** has real potential in {location} — small changes below can make a big difference.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Comparison vs city-wide average ----
        comp1, comp2 = st.columns(2)
        comp1.metric("Your Predicted Rating", f"{prediction:.2f} ⭐")
        comp2.metric("City-Wide Average", f"{global_mean:.2f} ⭐", delta=f"{delta_vs_avg:+.2f} vs avg")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading" style="text-align:left; font-size:1rem;">💡 What\'s Shaping This Rating</div>', unsafe_allow_html=True)

        for emoji, title, desc in insights:
            st.markdown(f"""
            <div class="glass-card" style="padding:1rem 1.3rem; margin-bottom:0.6rem;">
                <div style="display:flex; align-items:flex-start; gap:0.8rem;">
                    <div style="font-size:1.4rem;">{emoji}</div>
                    <div>
                        <div style="font-weight:700; color:#ffd166; font-family:'Poppins',sans-serif; font-size:0.95rem;">{title}</div>
                        <div style="color:rgba(255,255,255,0.7); font-size:0.85rem; margin-top:2px;">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ---- Welcoming landing view (shown before first prediction) ----
    st.markdown('<div class="section-heading">✨ Know Your Restaurant\'s Rating in 3 Simple Steps</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-emoji">📝</div>
            <div class="step-title">Enter Details</div>
            <div class="step-desc">Tell us your location, cuisine, cost, and services offered — in the panel on the left.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-emoji">🔮</div>
            <div class="step-title">Click Predict</div>
            <div class="step-desc">Our AI instantly analyzes your restaurant's profile against thousands of listings.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-emoji">⭐</div>
            <div class="step-title">Get Your Rating</div>
            <div class="step-desc">See your predicted rating on a live gauge, with instant feedback on where you stand.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🍲 Supports Every Cuisine & Restaurant Type</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chip-strip">
        <div class="chip">🍕 North Indian</div>
        <div class="chip">🍜 Chinese</div>
        <div class="chip">🍣 Continental</div>
        <div class="chip">🍔 Fast Food</div>
        <div class="chip">☕ Cafe</div>
        <div class="chip">🍰 Desserts</div>
        <div class="chip">🥗 South Indian</div>
        <div class="chip">🍻 Bar & Pub</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2.6rem;">👈</div>
            <p style="color:rgba(255,255,255,0.7); font-size:1rem; margin-top:0.3rem;">
                Start by filling in your restaurant's details in the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    🍽️ Restaurant Rating Predictor
</div>
""", unsafe_allow_html=True)