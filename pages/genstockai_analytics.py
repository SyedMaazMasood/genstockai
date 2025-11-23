# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import os
import time

# ==================== GEMINI AI SETUP ====================
"""
Google Gemini AI for Q&A Feature
- FREE tier with generous limits
- Fast responses
- Excellent at analyzing data and answering business questions
"""

try:
    import google.generativeai as genai
    
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Try different model names - Gemini API has different versions
        try:
            gemini_model = genai.GenerativeModel('gemini-pro')  # Stable model name
        except:
            try:
                gemini_model = genai.GenerativeModel('models/gemini-pro')  # Alternative format
            except:
                gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')  # Latest version
        
        GEMINI_ENABLED = True
    else:
        GEMINI_ENABLED = False
        gemini_model = None
except Exception as e:
    GEMINI_ENABLED = False
    gemini_model = None
    st.sidebar.warning(f"Gemini not available: {e}")

# ==================== EMBEDDED CONFIG & PROCESSOR ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")

def load_sales_data():
    if os.path.exists(SALES_DATA_FILE):
        with open(SALES_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

# Check AI providers
AI_PROVIDER = st.secrets.get("AI_PROVIDER", "groq").lower()

class CSVProcessor:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
    
    def _detect_columns(self):
        columns = self.df.columns.tolist()
        
        date_keywords = ['date', 'time', 'day', 'transaction']
        for col in columns:
            if any(kw in col for kw in date_keywords):
                self.column_mapping['date'] = col
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
                break
        
        product_keywords = ['product', 'item', 'name', 'description', 'sku']
        for col in columns:
            if any(kw in col for kw in product_keywords):
                self.column_mapping['product'] = col
                break
        
        quantity_keywords = ['quantity', 'qty', 'units', 'count', 'amount']
        for col in columns:
            if any(kw in col for kw in quantity_keywords) and 'price' not in col:
                self.column_mapping['quantity'] = col
                break
        
        price_keywords = ['price', 'cost', 'amount', 'total']
        for col in columns:
            if 'unit' in col or ('price' in col and 'total' not in col):
                self.column_mapping['unit_price'] = col
            elif 'total' in col or ('price' in col and 'total' in col):
                self.column_mapping['total_price'] = col
        
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
    
    def get_summary_stats(self):
        if self.df is None:
            return None
        
        stats = {
            'total_rows': len(self.df),
            'date_range': None,
            'unique_products': 0,
            'total_revenue': 0,
            'total_transactions': len(self.df)
        }
        
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            valid_dates = self.df[date_col].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'start': valid_dates.min(),
                    'end': valid_dates.max()
                }
        
        if 'product' in self.column_mapping:
            stats['unique_products'] = self.df[self.column_mapping['product']].nunique()
        
        if 'total_price' in self.column_mapping:
            stats['total_revenue'] = self.df[self.column_mapping['total_price']].sum()
        elif 'unit_price' in self.column_mapping and 'quantity' in self.column_mapping:
            self.df['calculated_total'] = (
                pd.to_numeric(self.df[self.column_mapping['unit_price']], errors='coerce') * 
                pd.to_numeric(self.df[self.column_mapping['quantity']], errors='coerce')
            )
            stats['total_revenue'] = self.df['calculated_total'].sum()
        
        return stats
    
    def analyze_product_performance(self):
        if self.df is None or 'product' not in self.column_mapping:
            return []
        
        product_col = self.column_mapping['product']
        quantity_col = self.column_mapping['quantity']
        
        product_analysis = self.df.groupby(product_col).agg({
            quantity_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        product_analysis.columns = ['product', 'total_quantity', 'transaction_count', 'avg_quantity']
        
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            date_range = (self.df[date_col].max() - self.df[date_col].min()).days
            weeks = max(date_range / 7, 1)
            product_analysis['weekly_velocity'] = product_analysis['total_quantity'] / weeks
        
        return product_analysis.to_dict('records')
    
    def detect_trends(self):
        if self.df is None:
            return {}
        
        trends = {
            'growing_products': [],
            'declining_products': []
        }
        
        if 'date' in self.column_mapping and 'product' in self.column_mapping:
            date_col = self.column_mapping['date']
            product_col = self.column_mapping['product']
            quantity_col = self.column_mapping['quantity']
            
            for product in self.df[product_col].unique():
                if pd.isna(product):
                    continue
                    
                product_data = self.df[self.df[product_col] == product].copy()
                product_data = product_data.sort_values(date_col)
                
                if len(product_data) >= 4:
                    mid_point = len(product_data) // 2
                    first_half = product_data.iloc[:mid_point][quantity_col].sum()
                    second_half = product_data.iloc[mid_point:][quantity_col].sum()
                    
                    if second_half > first_half * 1.2:
                        trends['growing_products'].append({
                            'product': product,
                            'growth_rate': ((second_half - first_half) / first_half * 100)
                        })
                    elif second_half < first_half * 0.8:
                        trends['declining_products'].append({
                            'product': product,
                            'decline_rate': ((first_half - second_half) / first_half * 100)
                        })
        
        return trends

# ==================== REAL GEMINI AI Q&A ====================
def generate_ai_response_with_gemini(question, stats, products, trends):
    """
    Uses REAL Google Gemini AI to answer questions about your data
    
    Gemini Pro:
    - FREE with generous limits (60 requests/min)
    - Fast responses (~1-2 seconds)
    - Excellent at data analysis and business intelligence
    - No credit card required for free tier
    """
    
    if not GEMINI_ENABLED:
        return "❌ **Gemini AI Not Configured**\n\nAdd GEMINI_API_KEY to secrets.toml to enable AI-powered Q&A.\n\nGet your free API key at: https://makersuite.google.com/app/apikey"
    
    try:
        # Prepare data summary for Gemini
        data_context = f"""
You are a business intelligence analyst helping a small business owner understand their sales data.

**Sales Data Summary:**
- Total Transactions: {stats.get('total_transactions', 0)}
- Unique Products: {stats.get('unique_products', 0)}
- Total Revenue: ${stats.get('total_revenue', 0):.2f}
- Data Time Range: {stats.get('date_range', {}).get('start', 'Unknown')} to {stats.get('date_range', {}).get('end', 'Unknown')}

**Top 5 Products by Sales:**
{chr(10).join([f"- {p['product']}: {int(p['total_quantity'])} units sold, {int(p['transaction_count'])} transactions, {p.get('weekly_velocity', 0):.1f} units/week" for p in sorted(products, key=lambda x: x['total_quantity'], reverse=True)[:5]])}

**Growth Trends:**
- Growing Products: {len(trends.get('growing_products', []))}
{chr(10).join([f"  - {p['product']}: +{p['growth_rate']:.1f}% growth" for p in trends.get('growing_products', [])[:3]])}

- Declining Products: {len(trends.get('declining_products', []))}
{chr(10).join([f"  - {p['product']}: -{p['decline_rate']:.1f}% decline" for p in trends.get('declining_products', [])[:3]])}

**User Question:** {question}

**Instructions:**
1. Answer the question based on the data above
2. Be specific and use actual numbers from the data
3. Provide actionable recommendations
4. Format your response with:
   - Clear sections with headers
   - Bullet points for key insights
   - Specific metrics and percentages
   - Confidence level (if applicable)
5. Keep response under 500 words
6. Use business-friendly language
"""
        
        # Call Gemini AI
        response = gemini_model.generate_content(data_context)
        
        return f"""**🤖 Gemini AI Analysis:**

{response.text}

---
**AI Model:** Google Gemini Pro (FREE)
**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Data Points:** {stats.get('total_transactions', 0):,} transactions"""
        
    except Exception as e:
        # Fallback if API fails
        return f"""**❌ Gemini API Error:**

{str(e)}

**Troubleshooting:**
1. Check that GEMINI_API_KEY is correctly set in secrets.toml
2. Verify API key is active at https://makersuite.google.com/
3. Check if you've hit rate limits (unlikely with free tier)

**Fallback:** Using template-based analysis..."""

# ==================== MAIN PAGE CODE ====================
st.title("📊 Analytics & Insights")
st.markdown("AI-powered analysis of your sales and inventory performance")

# Show AI status
col_ai1, col_ai2 = st.columns(2)
with col_ai1:
    if AI_PROVIDER == "groq":
        st.success("🤖 **Recommendations:** Groq Llama 3.1 (FREE)")
    elif AI_PROVIDER == "openai":
        st.info("🤖 **Recommendations:** OpenAI GPT-4o-mini (PAID)")
    else:
        st.warning("⚠️ **Recommendations:** Not configured")

with col_ai2:
    if GEMINI_ENABLED:
        st.success("🤖 **Q&A Analysis:** Google Gemini Pro (FREE)")
    else:
        st.warning("⚠️ **Q&A Analysis:** Add GEMINI_API_KEY")

st.markdown("---")

sales_data = load_sales_data()
inventory = load_inventory()

if not sales_data:
    st.warning("⚠️ No sales data available yet.")
    st.markdown("""
    **To view analytics:**
    1. Go to **Data Sources** page
    2. Upload your sales CSV file
    3. Process it with AI
    4. Return here to see detailed insights
    """)
    
    if st.button("📤 Upload Sales Data", type="primary"):
        st.switch_page("pages/genstockai_datasources.py")
    
    st.stop()

df = pd.DataFrame(sales_data)
processor = CSVProcessor()
processor.df = df

for col in processor.df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        try:
            processor.df[col] = pd.to_datetime(processor.df[col], errors='coerce')
        except:
            pass

processor._detect_columns()

stats = processor.get_summary_stats()
products = processor.analyze_product_performance()
trends = processor.detect_trends()

st.markdown("### 📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{stats.get('total_transactions', 0):,}")

with col2:
    st.metric("Unique Products", stats.get('unique_products', 0))

with col3:
    revenue = stats.get('total_revenue', 0)
    st.metric("Total Revenue", f"${revenue:,.2f}" if revenue else "N/A")

with col4:
    if stats.get('date_range'):
        days = (stats['date_range']['end'] - stats['date_range']['start']).days
        st.metric("Days of Data", days)
    else:
        st.metric("Days of Data", "N/A")

st.markdown("---")

st.markdown("### 🏆 Top Products by Sales")

if products:
    products_df = pd.DataFrame(products)
    products_df = products_df.sort_values('total_quantity', ascending=False).head(10)
    
    display_df = products_df.copy()
    display_df['total_quantity'] = display_df['total_quantity'].apply(lambda x: f"{int(x):,}")
    display_df['transaction_count'] = display_df['transaction_count'].apply(lambda x: f"{int(x):,}")
    
    if 'weekly_velocity' in display_df.columns:
        display_df['weekly_velocity'] = display_df['weekly_velocity'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product": "Product",
            "total_quantity": "Total Sold",
            "transaction_count": "# Transactions",
            "avg_quantity": "Avg per Transaction",
            "weekly_velocity": "Weekly Sales"
        }
    )
else:
    st.info("No product data available")

st.markdown("---")

st.markdown("### 📈 Trend Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🚀 Growing Products")
    growing = trends.get('growing_products', [])
    
    if growing:
        for item in growing[:5]:
            st.markdown(f"**{item['product']}**")
            st.progress(min(item['growth_rate'] / 100, 1.0))
            st.caption(f"↗️ +{item['growth_rate']:.1f}% growth")
            st.markdown("---")
    else:
        st.info("No significant growth trends detected")

with col2:
    st.markdown("#### 📉 Declining Products")
    declining = trends.get('declining_products', [])
    
    if declining:
        for item in declining[:5]:
            st.markdown(f"**{item['product']}**")
            st.progress(min(item['decline_rate'] / 100, 1.0))
            st.caption(f"↘️ -{item['decline_rate']:.1f}% decline")
            st.markdown("---")
    else:
        st.info("No significant decline trends detected")

st.markdown("---")

st.markdown("### 📦 Inventory Status")

if inventory:
    st.markdown(f"**Tracking {len(inventory)} products** (Updated from shelf scans)")
    
    inventory_data = []
    for product, data in inventory.items():
        if isinstance(data, dict):
            qty = data.get('quantity', 0)
            last_scanned = data.get('last_scanned', data.get('last_updated', 'N/A'))
            source = data.get('source', 'manual')
        else:
            qty = data
            last_scanned = 'N/A'
            source = 'manual'
        
        velocity = 0
        for p in products:
            if p['product'] == product:
                velocity = p.get('weekly_velocity', 0)
                break
        
        weeks_supply = (qty / velocity) if velocity > 0 else 999
        
        status = "✅ Good"
        if weeks_supply < 1:
            status = "🔴 Critical"
        elif weeks_supply < 2:
            status = "🟡 Low"
        
        inventory_data.append({
            'product': product,
            'quantity': qty,
            'weekly_velocity': velocity,
            'weeks_supply': weeks_supply,
            'status': status,
            'last_scanned': last_scanned,
            'source': source
        })
    
    inventory_data.sort(key=lambda x: x['weeks_supply'])
    
    for item in inventory_data[:10]:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{item['product']}**")
                if item['last_scanned'] != 'N/A':
                    st.caption(f"Last: {item['last_scanned'][:10]} ({item['source']})")
            
            with col2:
                st.markdown(f"{int(item['quantity'])} units")
            
            with col3:
                if item['weeks_supply'] < 999:
                    st.markdown(f"{item['weeks_supply']:.1f} weeks")
                else:
                    st.markdown("N/A")
            
            with col4:
                st.markdown(item['status'])

else:
    st.info("No inventory data. Use Shelf Scanner in Data Sources to update inventory.")

st.markdown("---")

# ==================== GEMINI AI Q&A SECTION ====================
st.markdown("### 🤖 AI-Powered Q&A")

with st.container(border=True):
    if GEMINI_ENABLED:
        st.markdown("#### Ask Google Gemini About Your Data")
        st.success("✅ **Powered by:** Google Gemini Pro (FREE)")
        st.caption("💡 Gemini analyzes your actual sales data and provides intelligent insights")
    else:
        st.markdown("#### AI Q&A Not Configured")
        st.warning("⚠️ Add GEMINI_API_KEY to secrets.toml to enable AI-powered analysis")
        st.markdown("""
**How to enable:**
1. Get free API key: https://makersuite.google.com/app/apikey
2. Add to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```
3. Restart app
""")
    
    user_question = st.text_input(
        "What would you like to know about your sales?",
        placeholder="e.g., What are my best-selling products? Which items should I promote? What trends do you see?",
        key="gemini_question"
    )
    
    col_btn1, col_btn2 = st.columns([1, 3])
    
    with col_btn1:
        ask_button = st.button("🧠 Ask Gemini AI", type="primary", disabled=not GEMINI_ENABLED, use_container_width=True)
    
    with col_btn2:
        if not GEMINI_ENABLED:
            st.caption("Add GEMINI_API_KEY to enable")
    
    if ask_button:
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("🤖 Gemini AI analyzing your data..."):
                ai_response = generate_ai_response_with_gemini(user_question, stats, products, trends)
                
                st.markdown("---")
                st.markdown(ai_response)

# Quick question suggestions
if GEMINI_ENABLED:
    st.markdown("**💡 Try asking:**")
    
    # 1. HELPER FUNCTION: This updates the text box safely
    def update_gemini_question(new_question):
        st.session_state['gemini_question'] = new_question

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Button 1: Profitable Products
        st.button(
            "💰 Which products are most profitable?", 
            use_container_width=True, 
            key="q1",
            on_click=update_gemini_question,
            args=("Which products are most profitable and should I focus on?",)
        )
    
    with col2:
        # Button 2: Declining Products
        st.button(
            "📉 What products are declining?", 
            use_container_width=True, 
            key="q2",
            on_click=update_gemini_question,
            args=("What products are declining and what should I do about them?",)
        )
    
    with col3:
        # Button 3: Strategy
        st.button(
            "🎯 What's my strategy?", 
            use_container_width=True, 
            key="q3",
            on_click=update_gemini_question,
            args=("Based on my sales data, what should my business strategy be?",)
        )

st.markdown("---")

st.markdown("### 📥 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export to CSV", use_container_width=True):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sales_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Product Report", use_container_width=True):
        if products:
            products_df = pd.DataFrame(products)
            csv = products_df.to_csv(index=False)
            st.download_button(
                label="Download Report",
                data=csv,
                file_name="product_report.csv",
                mime="text/csv"
            )

with col3:
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()

# Footer
st.markdown("---")
ai_stack = []
if AI_PROVIDER == "groq":
    ai_stack.append("Groq Llama 3.1 (Recommendations)")
elif AI_PROVIDER == "openai":
    ai_stack.append("OpenAI GPT-4o-mini (Recommendations)")

if GEMINI_ENABLED:
    ai_stack.append("Google Gemini Pro (Q&A)")

ai_text = " + ".join(ai_stack) if ai_stack else "Template-based Logic"
st.caption(f"🤖 Powered by: {ai_text} | Computer Vision: YOLOv8 + EasyOCR")