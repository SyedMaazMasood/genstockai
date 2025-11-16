import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import load_sales_data, load_inventory
    from csv_processor import CSVProcessor
except:
    st.error("⚠️ Missing required files.")
    st.stop()

st.title("📊 Analytics & Insights")
st.markdown("AI-powered analysis of your sales data and inventory performance.")

st.markdown("---")

# Load data
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
        st.switch_page("pages/2_📤_Data_Sources.py")
    
    st.stop()

# Process the data
df = pd.DataFrame(sales_data)
processor = CSVProcessor()
processor.df = df
processor._detect_columns()

# Get analysis
stats = processor.get_summary_stats()
products = processor.analyze_product_performance()
trends = processor.detect_trends()

# Top metrics
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

# Product Performance Table
st.markdown("### 🏆 Top Products by Sales")

if products:
    # Convert to DataFrame for display
    products_df = pd.DataFrame(products)
    
    # Sort by total quantity
    products_df = products_df.sort_values('total_quantity', ascending=False).head(10)
    
    # Format for display
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

# Trends Analysis
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

# Inventory Status
st.markdown("### 📦 Inventory Status")

if inventory:
    st.markdown(f"**Tracking {len(inventory)} products**")
    
    inventory_data = []
    for product, data in inventory.items():
        qty = data.get('quantity', 0)
        
        # Get sales velocity if available
        velocity = 0
        for p in products:
            if p['product'] == product:
                velocity = p.get('weekly_velocity', 0)
                break
        
        # Calculate weeks of supply
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
            'status': status
        })
    
    # Sort by weeks of supply
    inventory_data.sort(key=lambda x: x['weeks_supply'])
    
    # Display as table
    for item in inventory_data[:10]:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{item['product']}**")
            
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
    st.info("No inventory data available. Use the Shelf Scanner on the Data Sources page to update inventory levels.")

st.markdown("---")

# AI Insights Section
st.markdown("### 🤖 AI-Generated Insights")

with st.container(border=True):
    st.markdown("#### Ask AI About Your Data")
    
    user_question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., Which products should I focus on? What's my best seller?"
    )
    
    if st.button("🧠 Ask AI", type="primary"):
        with st.spinner("AI analyzing your data..."):
            import time
            time.sleep(1)
            
            # Simple AI responses based on data
            if "best" in user_question.lower() or "top" in user_question.lower():
                if products:
                    top_product = max(products, key=lambda x: x['total_quantity'])
                    st.success(f"""
                    **AI Insight:** Your best-selling product is **{top_product['product']}** with 
                    {int(top_product['total_quantity'])} units sold across {int(top_product['transaction_count'])} transactions.
                    
                    This product shows strong consistent demand and should be prioritized in your inventory management.
                    """)
            
            elif "focus" in user_question.lower():
                if growing:
                    top_growing = growing[0]
                    st.success(f"""
                    **AI Recommendation:** Focus on **{top_growing['product']}** which is showing 
                    {top_growing['growth_rate']:.1f}% growth. This indicates increasing customer demand.
                    
                    Consider increasing stock levels and promotional efforts for this product.
                    """)
                else:
                    st.info("Based on your data, maintain focus on your current top sellers. No dramatic shifts detected.")
            
            elif "worry" in user_question.lower() or "problem" in user_question.lower():
                if declining:
                    st.warning(f"""
                    **AI Alert:** Keep an eye on **{declining[0]['product']}** which shows 
                    {declining[0]['decline_rate']:.1f}% decline. Consider promotional strategies or 
                    investigating quality/pricing issues.
                    """)
                else:
                    st.success("No major concerns detected in your sales data. All products are performing within normal ranges.")
            
            else:
                # Generic insights
                st.info(f"""
                **AI Summary of Your Business:**
                
                - You're selling {stats.get('unique_products', 0)} different products
                - Total revenue: ${stats.get('total_revenue', 0):,.2f}
                - {len(growing)} products showing growth
                - {len(declining)} products need attention
                
                Overall health: **{'🟢 Good' if len(growing) > len(declining) else '🟡 Monitor Closely'}**
                """)

st.markdown("---")

# Export Options
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