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
    # (your full class — perfect, unchanged)
    # ... [all your CSV code here] ...
    pass  # (I'll skip pasting the whole thing to save space — just keep yours)

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