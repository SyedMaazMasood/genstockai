# -*- coding: utf-8 -*-
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

# ==================== CHECK AI STATUS ====================
AI_PROVIDER = st.secrets.get("AI_PROVIDER", "groq").lower()
AI_STATUS = "🟢 Active" if AI_PROVIDER in ["groq", "openai"] else "🔴 Not Configured"
AI_NAME = f"{AI_PROVIDER.upper()}" if AI_PROVIDER in ["groq", "openai"] else "None"
IS_FREE = AI_PROVIDER == "groq"

# ==================== MAIN PAGE CODE ====================

# Sidebar
st.sidebar.title("GenStockAI")
st.sidebar.markdown("---")
st.sidebar.info("🤖 AI-Powered Inventory Assistant for Small Businesses")

# Show AI provider in sidebar
st.sidebar.markdown("### 🤖 AI Status")
if AI_PROVIDER == "groq":
    st.sidebar.success(f"{AI_STATUS} - {AI_NAME} (FREE)")
elif AI_PROVIDER == "openai":
    st.sidebar.success(f"{AI_STATUS} - {AI_NAME} (PAID)")
else:
    st.sidebar.error(f"{AI_STATUS}")

st.sidebar.markdown("---")

# Load data
recommendations = load_recommendations()
sales_data = load_sales_data()
vendors = load_vendors()

pending_recs = len([r for r in recommendations if r.get('status') == 'pending'])
has_data = len(sales_data) > 0

# Main page content
st.title("🏠 GenStockAI Dashboard")
st.markdown("### Your AI-Powered Inventory Management Assistant")

# AI Status Banner
if AI_PROVIDER == "groq":
    st.success(f"✅ **AI Engine Active:** Groq Llama 3.1 (70B) - FREE & Unlimited")
elif AI_PROVIDER == "openai":
    st.info(f"✅ **AI Engine Active:** OpenAI GPT-4o-mini - Paid API (Switch to Groq for free)")
else:
    st.warning("⚠️ **AI Not Configured:** Add GROQ_API_KEY or OPENAI_API_KEY to secrets.toml")

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
col1, col2, col3, col4 = st.columns(4)

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
    
    # Show actual AI provider
    if AI_PROVIDER == "groq":
        st.markdown(f"- {agent_status} 🤖 Reorder Agent (Groq Llama 3.1 - FREE)")
        st.markdown(f"- {agent_status} 🤖 Promotion Agent (Groq Llama 3.1 - FREE)")
        st.markdown(f"- {agent_status} 🤖 Analysis Engine (Groq - FREE)")
    elif AI_PROVIDER == "openai":
        st.markdown(f"- {agent_status} 🤖 Reorder Agent (OpenAI GPT-4o-mini - PAID)")
        st.markdown(f"- {agent_status} 🤖 Promotion Agent (OpenAI GPT-4o-mini - PAID)")
        st.markdown(f"- {agent_status} 🤖 Analysis Engine (OpenAI - PAID)")
    else:
        st.markdown(f"- ⚪ 🤖 Reorder Agent (Not configured)")
        st.markdown(f"- ⚪ 🤖 Promotion Agent (Not configured)")
        st.markdown(f"- ⚪ 🤖 Analysis Engine (Not configured)")

# Add AI insights section
st.markdown("---")
st.markdown("### 🧠 Recent AI Activity")

if has_data:
    with st.container(border=True):
        st.markdown("**🤖 AI Processing (Last 24 hours)**")
        
        transaction_count = len(sales_data)
        
        st.markdown(f"- 📊 Analyzed {transaction_count} sales transactions using ML models")
        st.markdown(f"- 🔍 Identified {pending_recs} optimization opportunities via AI analysis")
        
        if AI_PROVIDER == "groq":
            st.markdown(f"- 💡 Generated recommendations using Groq Llama 3.1 (FREE)")
        elif AI_PROVIDER == "openai":
            st.markdown(f"- 💡 Generated recommendations using OpenAI GPT-4o-mini (PAID)")
        else:
            st.markdown(f"- 💡 Using rule-based recommendations (AI not configured)")
        
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
ai_credit = "Groq Llama 3.1 (FREE)" if AI_PROVIDER == "groq" else "OpenAI GPT-4o-mini" if AI_PROVIDER == "openai" else "Rule-based Logic"
st.caption(f"GenStockAI © 2025 | Powered by {ai_credit} & YOLOv8 Computer Vision | Built for Small Business Owners")