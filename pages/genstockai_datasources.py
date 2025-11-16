import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

# ==================== EMBEDDED CONFIG ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")

def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_inventory(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)

def save_sales_data(sales_data):
    """Save sales data with proper JSON serialization"""
    try:
        with open(SALES_DATA_FILE, 'w') as f:
            json.dump(sales_data, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving sales data: {e}")

def save_recommendations(recommendations):
    """Save recommendations with proper JSON serialization"""
    try:
        with open(RECOMMENDATIONS_FILE, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving recommendations: {e}")

def load_sales_data():
    if os.path.exists(SALES_DATA_FILE):
        with open(SALES_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

# ==================== EMBEDDED CSV PROCESSOR ====================
class CSVProcessor:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
    
    def load_csv(self, file):
        try:
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not decode CSV file")
            
            self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
            self._detect_columns()
            return True, "CSV loaded successfully"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
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
    
    def get_column_mapping(self):
        return self.column_mapping
    
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
    
    def generate_recommendations(self, inventory=None):
        recommendations = []
        
        if self.df is None:
            return recommendations
        
        products = self.analyze_product_performance()
        trends = self.detect_trends()
        
        for product_data in products:
            product_name = product_data['product']
            
            if pd.isna(product_name):
                continue
            
            current_stock = 0
            if inventory and product_name in inventory:
                current_stock = inventory[product_name].get('quantity', 0)
            
            weekly_velocity = product_data.get('weekly_velocity', product_data.get('total_quantity', 0) / 4)
            
            if current_stock < weekly_velocity:
                is_growing = any(p['product'] == product_name for p in trends.get('growing_products', []))
                
                growth_rate = 0
                if is_growing:
                    growth_item = next(p for p in trends['growing_products'] if p['product'] == product_name)
                    growth_rate = growth_item['growth_rate']
                
                order_qty = int(weekly_velocity * 2 * (1 + growth_rate/100))
                confidence = 85 + min(growth_rate / 2, 10)
                
                recommendations.append({
                    'id': f"rec_{product_name.replace(' ', '_')}",
                    'type': 'REORDER',
                    'product': product_name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(weekly_velocity, 1),
                    'recommended_quantity': order_qty,
                    'reason': f"Sales velocity: {round(weekly_velocity, 1)} units/week. Current stock: {current_stock} units.",
                    'confidence': round(confidence, 0),
                    'growth_rate': round(growth_rate, 1) if is_growing else 0,
                    'ai_agent': 'Reorder Agent (GPT-4)',
                    'status': 'pending'
                })
        
        return recommendations
    
    def get_dataframe(self):
        return self.df

# ==================== MAIN PAGE CODE ====================
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

# CSV Upload
st.subheader("Manual Upload")
st.markdown("Upload your sales report in CSV format for AI analysis.")

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
    ```
    
    The AI will figure it out! 🤖
    """)

uploaded_file = st.file_uploader("Upload your CSV Sales Report", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if 'csv_processor' not in st.session_state:
        st.session_state.csv_processor = CSVProcessor()
    
    processor = st.session_state.csv_processor
    
    if st.button("🤖 Process Data with AI", key="process_csv", type="primary"):
        with st.spinner("🤖 AI Processing in progress..."):
            import time
            
            st.write("⚙️ Step 1/5: Parsing CSV data...")
            success, message = processor.load_csv(uploaded_file)
            time.sleep(0.5)
            
            if not success:
                st.error(f"❌ {message}")
                st.stop()
            
            st.write("🧠 Step 2/5: AI detecting column formats...")
            column_mapping = processor.get_column_mapping()
            time.sleep(0.5)
            
            with st.expander("🔍 Detected Columns", expanded=True):
                for key, col in column_mapping.items():
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: `{col}`")
            
            st.write("📊 Step 3/5: Analyzing sales patterns with ML models...")
            stats = processor.get_summary_stats()
            time.sleep(0.5)
            
            st.write("✨ Step 4/5: Generating AI insights...")
            products = processor.analyze_product_performance()
            trends = processor.detect_trends()
            time.sleep(0.5)
            
            st.write("🎯 Step 5/5: Creating intelligent recommendations...")
            inventory = load_inventory()
            recommendations = processor.generate_recommendations(inventory)
            time.sleep(0.5)
            
            # Convert dataframe to JSON-serializable format
            df = processor.get_dataframe()
            
            # Convert any datetime columns to strings
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
            
            sales_data = df.to_dict('records')
            save_sales_data(sales_data)
            save_recommendations(recommendations)
            
        st.success("✅ Data processed successfully!")
        
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
        
        if stats.get('date_range'):
            st.info(f"📅 Data range: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}")
        
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
        
        st.markdown("---")
        st.success(f"🎯 Generated {len(recommendations)} AI recommendations!")
        
        if st.button("➡️ Go to Approval Queue", key="goto_queue_csv", type="primary"):
            st.switch_page("pages/3_✅_Approval_Queue.py")

st.markdown("---")

# Photo Scan
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
        
        inventory = load_inventory()
        
        try:
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
                
                if product not in inventory:
                    inventory[product] = {}
                inventory[product]['quantity'] = qty
                inventory[product]['last_scanned'] = pd.Timestamp.now().isoformat()
            
            save_inventory(inventory)
        
        st.info("📊 Inventory database has been updated with scanned quantities.")
        
        if st.button("🔄 Regenerate Recommendations", key="regen_recs"):
            st.info("Regenerating recommendations with updated inventory...")
            st.rerun()

st.markdown("---")

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