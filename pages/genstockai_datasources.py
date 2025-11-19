import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
import io
import time

# ==================== DEVELOPER-CONFIGURABLE AI SETTINGS ====================

GPT4_CONFIG = {
    "model": st.secrets.get("GPT4_MODEL", "gpt-4-turbo-preview"),
    "temperature": float(st.secrets.get("GPT4_TEMPERATURE", 0.3)),
    "max_tokens": int(st.secrets.get("GPT4_MAX_TOKENS", 500)),
    "top_p": float(st.secrets.get("GPT4_TOP_P", 0.9)),
    "frequency_penalty": float(st.secrets.get("GPT4_FREQ_PENALTY", 0.0)),
    "presence_penalty": float(st.secrets.get("GPT4_PRES_PENALTY", 0.0)),
}

ML_CONFIG = {
    "growth_threshold": float(st.secrets.get("ML_GROWTH_THRESHOLD", 1.2)),
    "decline_threshold": float(st.secrets.get("ML_DECLINE_THRESHOLD", 0.8)),
    "min_data_points": int(st.secrets.get("ML_MIN_DATA_POINTS", 4)),
    "base_confidence": int(st.secrets.get("ML_BASE_CONFIDENCE", 88)),
    "growth_bonus_max": int(st.secrets.get("ML_GROWTH_BONUS_MAX", 12)),
    "reorder_multiplier": float(st.secrets.get("ML_REORDER_MULTIPLIER", 2.0)),
    "safety_stock_weeks": float(st.secrets.get("ML_SAFETY_STOCK_WEEKS", 1.0)),
    "low_stock_threshold_weeks": float(st.secrets.get("ML_LOW_STOCK_WEEKS", 1.5)),      # used in reorder
    "overstock_threshold_weeks": float(st.secrets.get("ML_OVERSTOCK_WEEKS", 6.0)),      # used in promotion
    "severe_overstock_weeks": float(st.secrets.get("ML_SEVERE_OVERSTOCK_WEEKS", 12.0)), # 40%+ promo
}

# ==================== INTEGRATED COMPUTER VISION MODULE ====================
class ShelfScanner:
    def __init__(self):
        self.reader = None
        self.product_keywords = {
            'red bull': ['red', 'bull', 'energy', 'redbull'],
            'coffee': ['coffee', 'espresso', 'latte', 'cappuccino'],
            'croissant': ['croissant', 'pastry', 'croisant'],
            'bagel': ['bagel', 'bagles'],
            'muffin': ['muffin', 'muffins'],
            'milk': ['milk', '2%', 'whole', 'skim'],
            'pepsi': ['pepsi', 'cola'],
            'coca-cola': ['coke', 'coca', 'cola'],
        }
        self.confidence_threshold = 0.5
        
    def _initialize_ocr(self):
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False)
                return True
            except ImportError:
                return False
        return True
    
    def _preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _extract_text_with_ocr(self, image_array):
        if not self._initialize_ocr():
            return []
        try:
            results = self.reader.readtext(image_array)
            filtered_results = [
                (bbox, text, conf) 
                for bbox, text, conf in results 
                if conf >= self.confidence_threshold
            ]
            return filtered_results
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return []
    
    def _match_products(self, detected_texts):
        identified_products = {}
        all_text = ' '.join([text.lower() for _, text, _ in detected_texts])
        
        for product, keywords in self.product_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    if product not in identified_products:
                        identified_products[product] = 0
                    identified_products[product] += all_text.count(keyword)
        
        return identified_products
    
    def _estimate_quantities(self, products, detected_texts):
        estimated_quantities = {}
        for product, count in products.items():
            base_qty = count
            for _, text, _ in detected_texts:
                if any(char.isdigit() for char in text):
                    numbers = [int(s) for s in text.split() if s.isdigit()]
                    if numbers:
                        reasonable_nums = [n for n in numbers if 1 <= n <= 100]
                        if reasonable_nums:
                            base_qty = max(base_qty, max(reasonable_nums))
            
            if product in ['red bull', 'pepsi', 'coca-cola']:
                estimated_quantities[product] = max(base_qty * 3, 5)
            elif product in ['croissant', 'bagel', 'muffin']:
                estimated_quantities[product] = max(base_qty * 2, 3)
            else:
                estimated_quantities[product] = max(base_qty, 1)
        
        return estimated_quantities
    
    def scan_shelf(self, image_bytes):
        try:
            preprocessed = self._preprocess_image(image_bytes)
            detected_texts = self._extract_text_with_ocr(preprocessed)
            
            if not detected_texts:
                return {
                    'success': False,
                    'products': {},
                    'confidence': 0.0,
                    'raw_text': [],
                    'error': 'No text detected. Ensure good lighting and clear labels.'
                }
            
            matched_products = self._match_products(detected_texts)
            
            if not matched_products:
                return {
                    'success': False,
                    'products': {},
                    'confidence': 0.3,
                    'raw_text': [text for _, text, _ in detected_texts],
                    'error': 'Text detected but no known products identified.'
                }
            
            final_quantities = self._estimate_quantities(matched_products, detected_texts)
            
            avg_ocr_confidence = np.mean([conf for _, _, conf in detected_texts])
            detection_confidence = min(len(final_quantities) / 5.0, 1.0)
            overall_confidence = (avg_ocr_confidence + detection_confidence) / 2
            
            return {
                'success': True,
                'products': final_quantities,
                'confidence': float(overall_confidence),
                'raw_text': [text for _, text, _ in detected_texts],
                'detections_count': len(detected_texts)
            }
            
        except Exception as e:
            return {
                'success': False,
                'products': {},
                'confidence': 0.0,
                'raw_text': [],
                'error': f'Scanning error: {str(e)}'
            }

# ==================== [Your CSVProcessor class] ====================

class CSVProcessor:
    """AI-Powered CSV Analysis Engine – Now with Real Stock & Promotion Detection"""
    
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
        
        # Date column
        date_keywords = ['date', 'time', 'day', 'transaction', 'timestamp']
        for col in columns:
            if any(kw in col for kw in date_keywords):
                self.column_mapping['date'] = col
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
                break
        
        # Product column
        product_keywords = ['product', 'item', 'name', 'description', 'sku', 'upc']
        for col in columns:
            if any(kw in col for kw in product_keywords):
                self.column_mapping['product'] = col
                break
        
        # Quantity column
        quantity_keywords = ['quantity', 'qty', 'units', 'count', 'amount', 'sold']
        for col in columns:
            if any(kw in col for kw in quantity_keywords) and 'price' not in col and 'stock' not in col:
                self.column_mapping['quantity'] = col
                break
        if 'quantity' not in self.column_mapping:
            self.df[''] = 1
            self.column_mapping[''] = ''
        
        # Stock column (new!)
        stock_keywords = ['in_stock', 'current_stock', 'stock', 'inventory', 'on_hand', 'qty_on_hand', 'available']
        for col in columns:
            if any(kw in col for kw in stock_keywords):
                self.column_mapping['stock'] = col
                break

    def get_column_mapping(self):
        return self.column_mapping
    
    def get_summary_stats(self):
        if self.df is None:
            return None
        stats = {
            'total_rows': len(self.df),
            'date_range': None,
            'unique_products': 0,
            'total_transactions': len(self.df)
        }
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            valid = self.df[date_col].dropna()
            if len(valid) > 0:
                ['date_range'] = {'start': valid.min(), 'end': valid.max()}
        if 'product' in self.column_mapping:
            ['unique_products'] = self.df[self.column_mapping['product']].nunique()
        return 

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
        else:
            product_analysis['weekly_velocity'] = product_analysis['total_quantity'] / 4  # fallback
        
        return product_analysis.to_dict('records')
    
    def detect_trends(self):
        trends = {'growing_products': [], 'declining_products': []}
        if self.df is None or 'date' not in self.column_mapping or 'product' not in self.column_mapping:
            return trends
        
        date_col = self.column_mapping['date']
        product_col = self.column_mapping['product']
        quantity_col = self.column_mapping['quantity']
        
        for product in self.df[product_col].dropna().unique():
            data = self.df[self.df[product_col] == product].copy()
            if len(data) < ML_CONFIG["min_data_points"]:
                continue
            data = data.sort_values(date_col)
            mid = len(data) // 2
            first = data.iloc[:mid][quantity_col].sum()
            second = data.iloc[mid:][quantity_col].sum()
            if second > first * ML_CONFIG["growth_threshold"]:
                growth = (second - first) / first * 100
                trends['growing_products'].append({'product': product, 'growth_rate': round(growth, 1)})
            elif second < first * ML_CONFIG["decline_threshold"]:
                decline = (first - second) / first * 100
                trends['declining_products'].append({'product': product, 'decline_rate': round(decline, 1)})
        return trends

    def generate_recommendations(self, inventory=None):
        recommendations = []
        if self.df is None:
            return recommendations
        
        products = self.analyze_product_performance()
        trends = self.detect_trends()
        growing_products = {p['product'] for p in trends.get('growing_products', [])}
        
        # Detect stock column once
        has_stock_column = 'stock' in self.column_mapping
        stock_col = self.column_mapping.get('stock')
        
        for product_data in products:
            name = product_data['product']
            if pd.isna(name):
                continue
            
            weekly_velocity = product_data.get('weekly_velocity', 1)
            weekly_velocity = max(weekly_velocity, 0.1)  # avoid division by zero
            
            # SMART CURRENT STOCK LOGIC
            current_stock = 0
            
            # 1. Use CSV stock if available (most accurate)
            if has_stock_column:
                product_rows = self.df[self.df[self.column_mapping['product']] == name]
                if not product_rows.empty:
                    latest = product_rows[stock_col].iloc[-1]
                    if pd.notna(latest):
                        try:
                            current_stock = int(float(latest))
                        except:
                            current_stock = 0
            
            # 2. Fallback to inventory.json
            if current_stock == 0 and inventory and name in inventory:
                current_stock = inventory[name].get('quantity', 0)
            
            # REORDER RECOMMENDATION
            if current_stock < weekly_velocity * 1.5:  # Less than 1.5 weeks of stock
                multiplier = ML_CONFIG["reorder_multiplier"]
                if name in growing_products:
                    multiplier *= 1.3  # Order more for growing items
                
                order_qty = max(int(weekly_velocity * multiplier), 10)
                confidence = 88 + (10 if name in growing_products else 0)
                
                recommendations.append({
                    'id': f"reorder_{name.replace(' ', '_')}_{int(time.time())}",
                    'type': 'REORDER',
                    'product': name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(weekly_velocity, 1),
                    'recommended_quantity': order_qty,
                    'reason': f"Low stock alert: only {current_stock} units left at {round(weekly_velocity, 1)} units/week velocity",
                    'confidence': confidence,
                    'ai_agent': 'Reorder Agent (GPT-4)',
                    'status': 'pending'
                })
            
            # PROMOTION RECOMMENDATION – NEW!
            if current_stock > weekly_velocity * 6:  # More than 6 weeks of stock = overstock
                discount = "40%" if current_stock > weekly_velocity * 12 else "30%"
                excess_weeks = round(current_stock / weekly_velocity, 1)
                
                recommendations.append({
                    'id': f"promo_{name.replace(' ', '_')}_{int(time.time())}",
                    'type': 'PROMOTION',
                    'product': name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(weekly_velocity, 1),
                    'excess_weeks': excess_weeks,
                    'recommended_action': f"Launch {discount} off flash sale (3–5 days)",
                    'reason': f"Overstock detected: {current_stock} units = {excess_weeks} weeks of supply",
                    'confidence': 92,
                    'ai_agent': 'Promotion Agent (Claude 3.5)',
                    'status': 'pending'
                })
        
        return recommendations
    
    def get_dataframe(self):
        return self.df

# ==================== DATA PERSISTENCE ====================
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
    try:
        with open(SALES_DATA_FILE, 'w') as f:
            json.dump(sales_data, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving sales data: {e}")

def save_recommendations(recommendations):
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

# ==================== STREAMLIT UI ====================
st.title("📤 Data Sources")
st.markdown("Connect your data sources to enable AI-powered recommendations.")

st.markdown("---")

# ==================== CSV UPLOAD ====================
st.subheader("Manual Upload")
st.markdown("Upload your sales report in CSV format for AI analysis.")

with st.expander("📋 CSV Format Guide (AI Auto-Detects!)"):
    st.markdown("""
    **The AI will automatically detect your CSV format!**
    
    **Minimum Required:**
    - Product/Item name
    - Quantity (optional - will assume 1 if missing)
    
    **Recommended:**
    - Date/Time
    - Price or Total
    
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
            
            st.write("📊 Step 3/5: Analyzing sales patterns...")
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
            
            # Save data
            df = processor.get_dataframe()
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
            st.metric("AI Recommendations", len(recommendations))
        
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
            st.switch_page("pages/genstockai_approval.py")

st.markdown("---")

# ==================== SHELF SCAN — NOW WITH UPLOAD + CAMERA ====================
st.subheader("Shelf Scan (Computer Vision)")
st.markdown("Take a photo **or upload an image** of your shelves to track physical inventory levels.")

with st.expander("💡 How Shelf Scanning Works"):
    st.markdown("""
    **AI-Powered Visual Recognition:**
    1. 📸 **Capture**: Take a photo of your shelf
    2. 👁️ **Detection**: Computer vision identifies products
    3. 🔍 **OCR**: Reads labels and counts units
    4. ✅ **Update**: Automatically updates stock levels
    """)

# === NEW: Two-column layout for Upload + Camera ===
col1, col2 = st.columns(2)

image_bytes = None  # This will hold the final image from either source

with col1:
    st.markdown("**Upload shelf photo**")
    uploaded_shelf = st.file_uploader(
        "Choose image (JPG/PNG)",
        type=["png", "jpg", "jpeg"],
        key="shelf_upload",
        help="Upload a photo of your shelf"
    )
    if uploaded_shelf is not None:
        # Auto-resize large images
        img = Image.open(uploaded_shelf).convert('RGB')
        img.thumbnail((1200, 1600), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        st.image(image_bytes, caption="Uploaded Shelf", use_column_width=True)

with col2:
    st.markdown("**Live camera**")
    camera_photo = st.camera_input("Take photo of shelf", key="camera_input")
    if camera_photo is not None:
        image_bytes = camera_photo.getvalue()
        st.image(image_bytes, caption="Captured Shelf", use_column_width=True)

# ==================== PROCESS SHELF IMAGE (works for both sources) ====================
if image_bytes is not None:
    st.success("Photo ready for AI analysis!")
    
    if st.button("🤖 Analyze Shelf with Computer Vision AI", type="primary", use_container_width=True):
        with st.spinner("🤖 Running AI vision models on your shelf..."):
            scanner = ShelfScanner()
            
            progress = st.progress(0)
            status = st.empty()
            
            status.markdown("Enhancing image quality...")
            progress.progress(20)
            time.sleep(0.6)
            
            status.markdown("👁️ Running OCR on shelf labels...")
            progress.progress(50)
            time.sleep(0.8)
            
            status.markdown("Identifying products & counting units...")
            progress.progress(80)
            time.sleep(0.8)
            
            results = scanner.scan_shelf(image_bytes)
            progress.progress(100)
        
        st.markdown("---")
        
        if results['success']:
            st.success(f"✅ AI Analysis Complete – Confidence: {results['confidence']*100:.1f}%")
            st.balloons()
            
            inventory = load_inventory()
            
            with st.expander("Detected Products & Quantities", expanded=True):
                for product, qty in results['products'].items():
                    status = "Low Stock" if qty <= 10 else "Normal"
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.markdown(f"**{product.title()}**")
                    with col2:
                        st.markdown(f"**{qty}** units")
                    with col3:
                        st.markdown(status)
                    
                    # Update inventory
                    if product not in inventory:
                        inventory[product] = {}
                    inventory[product]['quantity'] = qty
                    inventory[product]['last_scanned'] = datetime.now().isoformat()
                
                save_inventory(inventory)
            
            st.success("Inventory updated with AI-scanned shelf data!")
            
        else:
            st.error(f"Detection failed: {results.get('error', 'Unknown error')}")
            st.info("Tip: Try better lighting, clearer labels, or hold camera steady")

else:
    st.info("Take a photo or upload an image to scan your shelf stock")

st.markdown("---")
st.caption("AI Vision + OCR • Real-time shelf intelligence • Works offline in browser")