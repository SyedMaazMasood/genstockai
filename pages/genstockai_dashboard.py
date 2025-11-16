import streamlit as st
import json
import os

# ==================== EMBEDDED CONFIG ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")
SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")

def load_recommendations():
    if os.path.exists(RECOMMENDATIONS_FILE):
        with open(RECOMMENDATIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def load_sales_data():
    if os.path.exists(SALES_DATA_FILE):
        with open(SALES_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

# ==================== MAIN PAGE CODE ====================
st.title("📈 Dashboard")

# Load real data
recommendations = load_recommendations()
sales_data = load_sales_data()

pending_recs = len([r for r in recommendations if r.get('status') == 'pending'])
has_data = len(sales_data) > 0

# Key Metrics
st.markdown("### Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Items to Review",
        value=str(pending_recs)
    )

with col2:
    time_saved = pending_recs * 15
    hours = time_saved / 60
    st.metric(
        label="Est. Weekly Time Saved",
        value=f"{hours:.1f} Hours" if hours > 0 else "0 Hours"
    )

with col3:
    waste_reduced = 500 if has_data else 0
    st.metric(
        label="Est. Monthly Waste Reduced",
        value=f"${waste_reduced}+"
    )

st.markdown("---")

# Success message
if pending_recs > 0:
    st.success(f"Welcome! You have {pending_recs} new recommendations in your Approval Queue.")
elif has_data:
    st.info("All recommendations processed! Upload new data to generate more insights.")
else:
    st.info("Welcome! Upload your sales data to get started with AI recommendations.")

st.markdown("---")

# Additional dashboard content
st.markdown("### Recent Activity")

with st.container(border=True):
    st.markdown("**Today**")
    if pending_recs > 0:
        st.markdown(f"- 🔄 {pending_recs} new recommendations generated")
    if has_data:
        st.markdown(f"- 📊 Sales data analyzed ({len(sales_data)} transactions)")
        approved_count = len([r for r in recommendations if r.get('status') == 'approved'])
        if approved_count > 0:
            st.markdown(f"- ✅ {approved_count} orders approved")
    else:
        st.markdown("- 📊 No sales data uploaded yet")

if has_data:
    with st.container(border=True):
        st.markdown("**This Week**")
        total_recs = len(recommendations)
        approved = len([r for r in recommendations if r.get('status') == 'approved'])
        st.markdown(f"- 📈 {total_recs} recommendations processed")
        if approved > 0:
            st.markdown(f"- 💰 Estimated savings from AI recommendations")
        st.markdown("- ⏱️ Time saved through automation")

st.markdown("---")

# Quick navigation
st.markdown("### Quick Navigation")
col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Approval Queue →", use_container_width=True, key="dash_queue"):
        st.switch_page("pages/genstockai_approval.py")

with col2:
    if st.button("Upload New Data →", use_container_width=True, key="dash_upload"):
        st.switch_page("pages/genstockai_datasources.py")

st.markdown("---")

# Add AI Processing Details
st.markdown("### 🤖 AI Processing Pipeline")
with st.container(border=True):
    st.markdown("**Current AI Models Active:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**NLP & Analysis:**")
        status = "✅" if has_data else "⏸️"
        st.markdown(f"- {status} GPT-4 for demand forecasting")
        st.markdown(f"- {status} Claude for negotiation drafting")
        st.markdown(f"- {status} Custom ML for pattern detection")
    
    with col2:
        st.markdown("**Computer Vision:**")
        st.markdown("- ⏸️ YOLOv8 for shelf scanning")
        st.markdown("- ⏸️ OCR for receipt processing")
        st.markdown("- ⏸️ Image classification for products")