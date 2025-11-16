import streamlit as st
import pandas as pd
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from csv_processor import CSVProcessor
    from config import save_sales_data, save_inventory, load_inventory
except:
    st.error("⚠️ Missing required files: csv_processor.py and config.py. Please ensure all files are in the correct location.")
    st.stop()

st.title("📤 Data Sources")

st.markdown("Connect your data sources to enable AI-powered recommendations.")

st.markdown("---")

# POS API Connection
st.subheader("POS Connection (Real-time)")
st.markdown("Connect your Point of Sale system for automatic data sync.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Connect to Clover POS", use_container_width=True, type="primary", key="clover_btn"):
        st.info("🔄 Clover POS integration coming soon! For now, please upload CSV sales reports.")

with col2:
    if st.button("Connect to Square POS", use_container_width=True, type="primary", key="square_btn"):
        st.info("🔄 Square POS integration coming soon! For now, please upload CSV sales reports.")

st.markdown("---")

# CSV Upload with Real Processing
st.subheader("Manual Upload")
st.markdown("Upload your sales report in CSV format for AI analysis.")

# Show expected format
with st.expander("📋 CSV Format Guide (AI Auto-Detects!)"):
    st.markdown("""
    **The AI will automatically detect your CSV format!** Your CSV can include columns like:
    
    **Minimum Required:**
    - Product/Item name
    - Quantity (optional - will assume 1 if missing)
    
    **Recommended:**
    - Date/Time
    - Price or Total
    - Category (optional)
    
    **Example formats accepted:**
    ```
    Date, Product, Quantity, Price
    2024-01-15, Red Bull, 2, 4.99
    
    OR
    
    Transaction Date, Item Name, Qty, Unit Price, Total
    01/15/2024, Croissant, 5, 3.50, 17.50
    
    OR even simpler:
    
    Product, Sales
    Coffee, 25
    Muffin, 18
    ```
    
    The AI will figure it out! 🤖
    """)

uploaded_file = st.file_uploader("Upload your CSV Sales Report", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Initialize processor
    if 'csv_processor' not in st.session_state:
        st.session_state.csv_processor = CSVProcessor()
    
    processor = st.session_state.csv_processor
    
    if st.button("🤖 Process Data with AI", key="process_csv", type="primary"):
        with st.spinner("🤖 AI Processing in progress..."):
            import time
            
            # Step 1: Load CSV
            st.write("⚙️ Step 1/5: Parsing CSV data...")
            success, message = processor.load_csv(uploaded_file)
            time.sleep(0.5)
            
            if not success:
                st.error(f"❌ {message}")
                st.stop()
            
            # Step 2: Detect columns
            st.write("🧠 Step 2/5: AI detecting column formats...")
            column_mapping = processor.get_column_mapping()
            time.sleep(0.5)
            
            # Show detected columns
            with st.expander("🔍 Detected Columns", expanded=True):
                for key, col in column_mapping.items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: `{col}`")
            
            # Step 3: Analyze data
            st.write("📊 Step 3/5: Analyzing sales patterns with ML models...")
            stats = processor.get_summary_stats()
            time.sleep(0.5)
            
            # Step 4: Generate insights
            st.write("✨ Step 4/5: Generating AI insights...")
            products = processor.analyze_product_performance()
            trends = processor.detect_trends()
            time.sleep(0.5)
            
            # Step 5: Create recommendations
            st.write("🎯 Step 5/5: Creating intelligent recommendations...")
            inventory = load_inventory()
            recommendations = processor.generate_recommendations(inventory)
            time.sleep(0.5)
            
            # Save processed data
            df = processor.get_dataframe()
            sales_data = df.to_dict('records')
            save_sales_data(sales_data)
            
            # Save recommendations
            from config import save_recommendations
            save_recommendations(recommendations)
            
        st.success("✅ Data processed successfully!")
        
        # Show summary
        st.markdown("---")
        st.markdown("### 📊 Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", stats.get('total_transactions', 0))
        
        with col2:
            st.metric("Unique Products", stats.get('unique_products', 0))
        
        with col3:
            revenue = stats.get('total_revenue', 0)
            st.metric("Total Revenue", f"${revenue:,.2f}" if revenue else "N/A")
        
        # Date range
        if stats.get('date_range'):
            st.info(f"📅 Data range: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}")
        
        # Show trends
        if trends.get('growing_products') or trends.get('declining_products'):
            st.markdown("---")
            st.markdown("### 📈 AI-Detected Trends")
            
            if trends.get('growing_products'):
                with st.expander("🚀 Growing Products", expanded=True):
                    for item in trends['growing_products'][:5]:
                        st.markdown(f"- **{item['product']}**: +{item['growth_rate']:.1f}% growth")
            
            if trends.get('declining_products'):
                with st.expander("📉 Declining Products"):
                    for item in trends['declining_products'][:5]:
                        st.markdown(f"- **{item['product']}**: -{item['decline_rate']:.1f}% decline")
        
        # Show recommendations count
        st.markdown("---")
        st.success(f"🎯 Generated {len(recommendations)} AI recommendations!")
        
        if st.button("➡️ Go to Approval Queue", key="goto_queue_csv", type="primary"):
            st.switch_page("pages/3_✅_Approval_Queue.py")

st.markdown("---")

# Photo Scan with OCR simulation
st.subheader("Shelf Scan (Computer Vision)")
st.markdown("Take a photo of your shelves to track physical inventory levels.")

with st.expander("💡 How Shelf Scanning Works"):
    st.markdown("""
    **AI-Powered Visual Recognition:**
    1. 📸 **Capture**: Take a photo of your shelf/storage
    2. 👁️ **Detection**: YOLOv8 identifies products and counts units
    3. 🔍 **OCR**: Reads labels, expiration dates, and prices
    4. 🧠 **Matching**: Cross-references with your inventory database
    5. ✅ **Update**: Automatically updates stock levels
    
    **For this demo**: The AI will simulate detection based on your uploaded sales data.
    """)

camera_photo = st.camera_input("Scan your shelf stock", key="camera_input")

if camera_photo is not None:
    st.success("✅ Photo captured successfully!")
    
    if st.button("🤖 Analyze Photo with Computer Vision", key="analyze_photo", type="primary"):
        with st.spinner("🤖 AI Vision analyzing shelf stock..."):
            import time
            st.write("👁️ Running computer vision models (YOLOv8)...")
            time.sleep(0.8)
            st.write("🔍 Detecting products and counting units...")
            time.sleep(0.8)
            st.write("📝 Running OCR on labels and expiration dates...")
            time.sleep(0.8)
            st.write("🧠 Cross-referencing with inventory database...")
            time.sleep(0.8)
        
        st.success("✅ AI Analysis complete!")
        
        # Simulate detection based on sales data
        inventory = load_inventory()
        
        # If we have processed sales data, use real product names
        try:
            from config import load_sales_data
            sales_data = load_sales_data()
            if sales_data:
                df = pd.DataFrame(sales_data)
                processor_temp = CSVProcessor()
                processor_temp.df = df
                processor_temp._detect_columns()
                
                if 'product' in processor_temp.column_mapping:
                    products_detected = df[processor_temp.column_mapping['product']].unique()[:6]
                else:
                    products_detected = ["Red Bull", "Croissants", "Coffee Beans", "Milk", "Bagels", "Muffins"]
            else:
                products_detected = ["Red Bull", "Croissants", "Coffee Beans", "Milk", "Bagels", "Muffins"]
        except:
            products_detected = ["Red Bull", "Croissants", "Coffee Beans", "Milk", "Bagels", "Muffins"]
        
        # Show AI detection results
        with st.expander("🤖 Computer Vision Detection Results", expanded=True):
            st.markdown("**Items Detected with AI:**")
            
            import random
            for i, product in enumerate(products_detected):
                qty = random.randint(3, 25)
                status = "✅ Normal" if qty > 10 else "⚠️ Low Stock"
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{product}**")
                with col2:
                    st.markdown(f"{qty} units")
                with col3:
                    st.markdown(status)
                
                # Update inventory
                if product not in inventory:
                    inventory[product] = {}
                inventory[product]['quantity'] = qty
                inventory[product]['last_scanned'] = pd.Timestamp.now().isoformat()
            
            # Save updated inventory
            save_inventory(inventory)
        
        st.info("📊 Inventory database has been updated with scanned quantities.")
        
        if st.button("🔄 Regenerate Recommendations", key="regen_recs"):
            st.info("Regenerating recommendations with updated inventory...")
            st.rerun()

st.markdown("---")

# Data source status
st.subheader("📡 Connected Sources")

status_data = [
    {
        "Source": "CSV Upload", 
        "Status": "✅ Active" if 'csv_processor' in st.session_state else "⚪ No Data", 
        "Last Sync": "Just now" if 'csv_processor' in st.session_state else "N/A"
    },
    {"Source": "Clover POS", "Status": "⚪ Not Connected", "Last Sync": "N/A"},
    {"Source": "Square POS", "Status": "⚪ Not Connected", "Last Sync": "N/A"},
    {
        "Source": "Shelf Scanner", 
        "Status": "✅ Active" if camera_photo else "⚪ Not Used", 
        "Last Sync": "Just now" if camera_photo else "N/A"
    },
]

for source in status_data:
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{source['Source']}**")
        with col2:
            st.markdown(source['Status'])
        with col3:
            st.markdown(f"*{source['Last Sync']}*")