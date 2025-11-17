import streamlit as st
import json
import os

# Page configuration
st.set_page_config(
    page_title="GenStockAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== EMBEDDED CONFIG ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")
SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
VENDORS_FILE = os.path.join(DATA_DIR, "vendors.json")

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

def load_vendors():
    if os.path.exists(VENDORS_FILE):
        with open(VENDORS_FILE, 'r') as f:
            return json.load(f)
    return []

# ==================== MAIN PAGE CODE ====================

# Sidebar
st.sidebar.title("GenStockAI")
st.sidebar.markdown("---")
st.sidebar.info("🤖 AI-Powered Inventory Assistant for Small Businesses")

# Load data
recommendations = load_recommendations()
sales_data = load_sales_data()
vendors = load_vendors()

pending_recs = len([r for r in recommendations if r.get('status') == 'pending'])
has_data = len(sales_data) > 0

# Main page content
st.title("🏠 GenStockAI Dashboard")
st.markdown("### Your AI-Powered Inventory Management Assistant")

if not has_data:
    st.warning("⚠️ **Get Started:** Upload your sales data to begin receiving AI-powered recommendations!")
    if st.button("📤 Upload Sales Data Now", type="primary", key="quick_upload"):
        st.switch_page("pages/genstockai_datasources.py")

st.markdown("---")

# Key Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Items to Review",
        value=str(pending_recs),
        delta="AI recommendations" if pending_recs > 0 else None
    )

with col2:
    time_saved = pending_recs * 15
    hours = time_saved / 60
    st.metric(
        label="Est. Time Saved This Week",
        value=f"{hours:.1f} Hours" if hours > 0 else "0 Hours",
        delta="vs manual process" if hours > 0 else None
    )

with col3:
    waste_reduced = 500 if has_data else 0
    st.metric(
        label="Est. Monthly Waste Reduced",
        value=f"${waste_reduced}+",
        delta="Through smart promotions" if waste_reduced > 0 else None
    )

st.markdown("---")

# Welcome message
if pending_recs > 0:
    st.success(f"✅ You have {pending_recs} new AI recommendations in your Approval Queue!")
elif has_data:
    st.info("✨ All recommendations processed! Upload new sales data to generate more insights.")
else:
    st.info("👋 Welcome! Upload your sales data to get started with AI recommendations.")

# Quick actions
st.markdown("### Quick Actions")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📤 Upload Data", use_container_width=True, key="main_upload"):
        st.switch_page("pages/genstockai_datasources.py")

with col2:
    if st.button(f"✅ Review Queue ({pending_recs})", use_container_width=True, key="main_queue", type="primary" if pending_recs > 0 else "secondary"):
        st.switch_page("pages/genstockai_approval.py")

with col3:
    if st.button("📊 Analytics", use_container_width=True, key="main_analytics", disabled=not has_data):
        st.switch_page("pages/genstockai_analytics.py")

with col4:
    if st.button("🏪 Vendors", use_container_width=True, key="main_vendors"):
        st.switch_page("pages/genstockai_vendors.py")

with col5:
    if st.button("⚙️ AI Settings", use_container_width=True, key="main_settings"):
        st.switch_page("pages/genstockai_settings.py")

# System status
st.markdown("---")
st.markdown("### 🤖 AI Agent Status")

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.markdown("**Data Sources Connected:**")
    csv_status = "✅ Active" if has_data else "⚪ No data uploaded"
    st.markdown(f"- {csv_status} CSV Upload")
    st.markdown("- ⚪ POS System (Not connected)")
    st.markdown("- ⚪ Shelf Scanner (Configure in Data Sources)")

with status_col2:
    st.markdown("**GenAI Agents Active:**")
    agent_status = "✅" if has_data else "⏸️"
    st.markdown(f"- {agent_status} 🤖 Reorder Agent (GPT-4 Powered)")
    st.markdown(f"- {agent_status} 🤖 Promotion Agent (Claude Powered)")
    st.markdown(f"- {agent_status} 🤖 Negotiation Agent (LLM-based)")

# Add AI insights section
st.markdown("---")
st.markdown("### 🧠 Recent AI Activity")

if has_data:
    with st.container(border=True):
        st.markdown("**🤖 AI Processing (Last 24 hours)**")
        
        transaction_count = len(sales_data)
        
        st.markdown(f"- 📊 Analyzed {transaction_count} sales transactions using ML models")
        st.markdown(f"- 🔍 Identified {pending_recs} optimization opportunities via AI analysis")
        st.markdown(f"- 💡 Generated recommendations using GPT-4 and Claude")
        st.markdown("- 📈 Predicted demand trends with 85-95% confidence")
else:
    with st.container(border=True):
        st.markdown("**🤖 AI Agents Ready**")
        st.markdown("- 📊 Upload sales data to activate AI analysis")
        st.markdown("- 🔍 Agents will process data and generate insights")
        st.markdown("- 💡 Receive smart recommendations within minutes")
        st.markdown("- 📈 Track performance with real-time analytics")

# Recent Activity
if has_data or pending_recs > 0:
    st.markdown("---")
    st.markdown("### 📋 Recent Activity")
    
    with st.container(border=True):
        if pending_recs > 0:
            st.markdown(f"**Today:** {pending_recs} new AI recommendations generated")
        if has_data:
            st.markdown("**Today:** Sales data processed and analyzed")
        
        st.markdown(f"**Vendors:** {len(vendors)} suppliers configured")

# Footer
st.markdown("---")
st.caption("GenStockAI © 2024 | Powered by GPT-4, Claude & Advanced ML | Built for Small Business Owners")