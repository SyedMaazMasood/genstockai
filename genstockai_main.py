import streamlit as st

# Page configuration
st.set_page_config(
    page_title="GenStockAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("GenStockAI")
st.sidebar.markdown("---")
st.sidebar.info("🤖 AI-Powered Inventory Assistant for Small Businesses")

# Main page content
st.title("🏠 GenStockAI Dashboard")
st.markdown("### Your AI-Powered Inventory Management Assistant")

st.markdown("---")

# Key Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Items to Review",
        value="3",
        delta="New recommendations"
    )

with col2:
    st.metric(
        label="Est. Weekly Time Saved",
        value="7 Hours",
        delta="vs manual process"
    )

with col3:
    st.metric(
        label="Est. Monthly Waste Reduced",
        value="$500+",
        delta="Through smart promotions"
    )

st.markdown("---")

# Welcome message
st.success("✅ Welcome! You have 3 new recommendations in your Approval Queue.")

# Quick actions
st.markdown("### Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📤 Upload Sales Data", use_container_width=True, key="main_upload"):
        st.switch_page("pages/2_📤_Data_Sources.py")

with col2:
    if st.button("✅ Review Queue", use_container_width=True, key="main_queue"):
        st.switch_page("pages/3_✅_Approval_Queue.py")

with col3:
    st.button("📊 View Analytics", use_container_width=True, disabled=True, key="main_analytics")
    st.caption("Coming soon")

# System status
st.markdown("---")
st.markdown("### 🤖 AI Agent Status")

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.markdown("**Data Sources Connected:**")
    st.markdown("- ✅ CSV Upload (Last: 2 days ago)")
    st.markdown("- ⚪ POS System (Not connected)")
    st.markdown("- ⚪ Shelf Scanner (Not configured)")

with status_col2:
    st.markdown("**GenAI Agents Active:**")
    st.markdown("- ✅ 🤖 Reorder Agent (GPT-4 Powered)")
    st.markdown("- ✅ 🤖 Promotion Agent (Claude Powered)")
    st.markdown("- ✅ 🤖 Negotiation Agent (LLM-based)")

# Add AI insights section
st.markdown("---")
st.markdown("### 🧠 Recent AI Insights")
with st.container(border=True):
    st.markdown("**🤖 AI Agent Activity (Last 24 hours)**")
    st.markdown("- 📊 Analyzed 247 sales transactions using ML models")
    st.markdown("- 🔍 Identified 3 optimization opportunities via NLP analysis")
    st.markdown("- 💡 Generated supplier negotiation templates using GPT-4")
    st.markdown("- 📈 Predicted next week's demand with 94% accuracy")

# Footer
st.markdown("---")
st.caption("GenStockAI © 2024 | Powered by AI | Built for Small Business Owners")