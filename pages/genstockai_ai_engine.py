# -*- coding: utf-8 -*-
import streamlit as st
import time
import os

# ==================== 1. SETUP AI CLIENT ====================
# We need to initialize the client to make real calls
AI_PROVIDER = st.secrets.get("AI_PROVIDER", "groq").lower()
ai_client = None
AI_MODEL = "llama-3.3-70b-versatile" # Default free model

try:
    if AI_PROVIDER == "groq":
        from groq import Groq
        if "GROQ_API_KEY" in st.secrets:
            ai_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            AI_MODEL = "llama-3.3-70b-versatile"
    elif AI_PROVIDER == "openai":
        from openai import OpenAI
        if "OPENAI_API_KEY" in st.secrets:
            ai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            AI_MODEL = "gpt-4o-mini"
except Exception as e:
    st.error(f"Error initializing AI: {e}")

# ==================== HELPER: REAL AI GENERATION ====================
def generate_real_analysis(system_role, user_prompt):
    """Sends a request to the active AI provider and returns the text."""
    if not ai_client:
        return "⚠️ AI Client not initialized. Check API Keys."
    
    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3, # Low temperature for consistent facts
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# ==================== MAIN PAGE UI ====================
st.title("🤖 AI Engine")
st.markdown("### See How GenAI Powers Your Inventory Management")

# Show active AI provider banner
if AI_PROVIDER == "groq":
    st.success(f"✅ **Active AI:** Groq ({AI_MODEL}) - FREE & Unlimited")
elif AI_PROVIDER == "openai":
    st.success(f"✅ **Active AI:** OpenAI ({AI_MODEL}) - PAID")
else:
    st.error("❌ **No AI Configured:** Add GROQ_API_KEY to secrets.toml")

st.markdown("---")

# AI Architecture Overview
st.markdown("## 🗂️ Multi-Agent AI Architecture")

with st.container(border=True):
    st.markdown("### **Three Specialized AI Agents Working for You:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔄 Reorder Agent")
        st.markdown(f"**Powered by:** {AI_MODEL}")
        st.markdown("**Function:** Analyzes sales velocity vs stock")
        st.markdown("**Output:** Precise reorder quantities")
    
    with col2:
        st.markdown("#### 🏷️ Promotion Agent")
        st.markdown(f"**Powered by:** {AI_MODEL}")
        st.markdown("**Function:** Detects overstock & expiry risks")
        st.markdown("**Output:** Discount strategies")
    
    with col3:
        st.markdown("#### 💬 Analysis Engine")
        st.markdown(f"**Powered by:** {AI_MODEL}")
        st.markdown("**Function:** Business Intelligence Q&A")
        st.markdown("**Output:** Strategic insights")

st.markdown("---")

# ==================== LIVE AI DEMO (REAL CALLS) ====================
st.markdown("## 🎬 Live AI Demo")
st.markdown("Watch our AI agents analyze **Sample Data** in real-time:")

if st.button("▶️ Run Live AI Analysis", type="primary", use_container_width=True, key="run_demo"):
    
    if not ai_client:
        st.error("Please configure API keys in .streamlit/secrets.toml first!")
        st.stop()

    # --- AGENT 1: REORDER AGENT ---
    with st.container(border=True):
        st.markdown("### 🔄 Reorder Agent - Analysis")
        
        # 1. Simulate "Loading" (Visual effect only)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("**📥 Ingesting sample data: Red Bull Energy Drink...**")
        time.sleep(0.5)
        progress_bar.progress(30)
        
        # 2. DEFINE THE DATA (We send this to the AI)
        sample_data_context = """
        Product: Red Bull 8oz
        Current Stock: 8 units
        Avg Weekly Sales: 36 units
        Lead Time: 2 days
        Trend: +20% sales spike on Fridays
        Safety Stock Requirement: 12 units
        """
        
        # 3. CALL REAL AI
        status_text.markdown(f"**🧠 {AI_MODEL} is thinking...**")
        progress_bar.progress(60)
        
        system_prompt = "You are an inventory optimization AI. Analyze the data and recommend a specific reorder quantity. Be concise. Format: Analysis -> Recommendation -> Confidence Score."
        
        # This is the actual API call
        ai_response = generate_real_analysis(system_prompt, f"Analyze this inventory status: {sample_data_context}")
        
        progress_bar.progress(100)
        status_text.empty()
        
        # 4. Display Result
        st.success("**AI Recommendation Generated**")
        st.markdown(f"**Input Data:** {sample_data_context}")
        
        with st.expander("🧠 View AI Reasoning (Real Output)", expanded=True):
            st.markdown(ai_response)

    # --- AGENT 2: PROMOTION AGENT ---
    with st.container(border=True):
        st.markdown("### 🏷️ Promotion Agent - Analysis")
        
        # 1. Visual Loading
        progress_bar2 = st.progress(0)
        status_text2 = st.empty()
        status_text2.markdown("**📥 Ingesting sample data: Fresh Croissants...**")
        time.sleep(0.5)
        progress_bar2.progress(30)
        
        # 2. DEFINE DATA
        sample_promo_data = """
        Product: Butter Croissants
        Current Stock: 20 units (Normal daily sales is only 8)
        Shelf Life: Expires in 24 hours
        Current Time: 2:00 PM
        Cost per unit: $1.00
        Selling Price: $3.50
        """
        
        # 3. CALL REAL AI
        status_text2.markdown(f"**🧠 {AI_MODEL} is thinking...**")
        progress_bar2.progress(60)
        
        promo_system = "You are a retail pricing AI. Given the overstock/expiry risk, suggest a specific promotion to clear stock without losing money. Be concise."
        
        ai_response_2 = generate_real_analysis(promo_system, f"Analyze this situation and create a promo: {sample_promo_data}")
        
        progress_bar2.progress(100)
        status_text2.empty()
        
        # 4. Display Result
        st.success("**AI Strategy Generated**")
        st.markdown(f"**Input Data:** {sample_promo_data}")
        
        with st.expander("🧠 View AI Reasoning (Real Output)", expanded=True):
            st.markdown(ai_response_2)

# --- AGENT 3: NEGOTIATION AGENT ---
    with st.container(border=True):
        st.markdown("### 💬 Negotiation Agent - Analysis")
        
        # 1. Visual Loading
        progress_bar3 = st.progress(0)
        status_text3 = st.empty()
        status_text3.markdown("**📥 Ingesting supplier data: Peak Coffee...**")
        time.sleep(0.5)
        progress_bar3.progress(20)
        
        # 2. DEFINE DATA (We send this to the AI)
        sample_negotiation_data = """
        Vendor: Peak Coffee
        Product: Premium Arabica Beans
        Current Price: $12.50/lb
        Monthly Volume: 40 lbs
        Competitor Offer (Roast Co.): $11.88/lb
        Target Discount: 5% ($0.62/lb reduction)
        Relationship Duration: 8 months
        Payment History: 100% On-time
        """
        
        # 3. CALL REAL AI
        # We update the progress bar while "thinking"
        steps3 = [
            "Web scraping competitor prices (Simulated)...",
            "Analyzing supplier payment history...",
            f"🧠 {AI_MODEL} drafting negotiation strategy...",
            "Calculating projected annual savings..."
        ]
        
        for i, step in enumerate(steps3):
            status_text3.markdown(f"**{step}**")
            progress_bar3.progress(20 + (i+1)*15)
            time.sleep(0.5)
            
        negotiation_system = """
        You are a procurement expert. Write a professional negotiation email to a vendor. 
        Use the provided data to justify a price reduction request. 
        Tone: Professional, appreciative of the partnership, but firm on the market price reality.
        Include a subject line.
        """
        
        # Actual API Call
        ai_response_3 = generate_real_analysis(negotiation_system, f"Write a negotiation email based on this data: {sample_negotiation_data}")
        
        progress_bar3.progress(100)
        status_text3.empty()
        
        # 4. Display Result
        st.success("**AI Recommendation Generated:** Negotiate 5% discount with Peak Coffee")
        st.markdown(f"**Input Data:** {sample_negotiation_data}")
        
        with st.expander("🧠 View AI-Generated Email (Real Output)", expanded=True):
            st.markdown(ai_response_3)
            st.caption(f"💡 Generated dynamically by {AI_MODEL}")

st.markdown("---")

# Technology Stack
st.markdown("## 🔧 Technology Stack")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("### **Generative AI**")
        st.markdown(f"- **Model:** {AI_MODEL}")
        st.markdown("- **Role:** Decision making, reasoning, strategy")
        st.markdown(f"- **Provider:** {AI_PROVIDER.upper()}")
with col2:
    with st.container(border=True):
        st.markdown("### **Agents & Tools**")
        st.markdown("- **Reorder Agent:** Time-series analysis")
        st.markdown("- **Promo Agent:** Revenue optimization logic")
        st.markdown("- **Negotiation Agent:** NLP & persuasion engines")

st.markdown("---")

# Footer
st.caption(f"🤖 Powered by {AI_PROVIDER.upper()} | Live Analysis")