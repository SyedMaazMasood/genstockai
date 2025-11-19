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

# ==================== DEVELOPER CONFIGURATION ====================
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
    "low_stock_threshold_weeks": float(st.secrets.get("ML_LOW_STOCK_WEEKS", 1.5)),
    "overstock_threshold_weeks": float(st.secrets.get("ML_OVERSTOCK_WEEKS", 6.0)),
    "severe_overstock_weeks": float(st.secrets.get("ML_SEVERE_OVERSTOCK_WEEKS", 12.0)),
}

# ==================== DATA PERSISTENCE (MUST BE BEFORE ANYTHING USES IT) ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")

def save_sales_data(sales_records):
    """Save sales data as list of dicts"""
    try:
        with open(SALES_DATA_FILE, 'w') as f:
            json.dump(sales_records, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving sales data: {e}")

def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_inventory(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)

def save_recommendations(recommendations):
    with open(RECOMMENDATIONS_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)

# ==================== SHELF SCANNER (unchanged) ====================
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
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
    
    def scan_shelf(self, image_bytes):
        try:
            preprocessed = self._preprocess_image(image_bytes)
            if not self._initialize_ocr():
                return {'success': False, 'error': 'EasyOCR not available'}
            results = self.reader.readtext(preprocessed)
            texts = [(b, t, c) for b, t, c in results if c >= self.confidence_threshold]
            if not texts:
                return {'success': False, 'error': 'No text detected'}
            all_text = ' '.join([t.lower() for _, t, _ in texts])
            matched = {}
            for prod, kws in self.product_keywords.items():
                if any(kw in all_text for kw in kws):
                    matched[prod] = matched.get(prod, 0) + sum(all_text.count(kw) for kw in kws)
            if not matched:
                return {'success': False, 'error': 'No known products'}
            quantities = {}
            for p, c in matched.items():
                qty = max(c * 3 if p in ['red bull','pepsi','coca-cola'] else c * 2, 5)
                quantities[p] = qty
            return {'success': True, 'products': quantities, 'confidence': 0.8}
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ==================== CSV PROCESSOR ====================
class CSVProcessor:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
    
    def load_csv(self, file):
        try:
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(file, encoding=enc)
                    break
                except:
                    continue
            if self.df is None:
                return False, "Cannot read CSV"
            self.df.columns = [c.strip().lower().replace(' ', '_') for c in self.df.columns]
            self._detect_columns()
            return True, "Loaded"
        except Exception as e:
            return False, str(e)
    
    def _detect_columns(self):
        cols = self.df.columns
        for kw in ['date', 'time', 'transaction']: 
            for c in cols:
                if kw in c:
                    self.column_mapping['date'] = c
                    self.df[c] = pd.to_datetime(self.df[c], errors='coerce')
        for kw in ['product', 'item', 'name', 'description', 'sku']:
            for c in cols:
                if kw in c:
                    self.column_mapping['product'] = c
        for kw in ['quantity', 'qty', 'units', 'sold']:
            for c in cols:
                if kw in c and 'price' not in c:
                    self.column_mapping['quantity'] = c
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
    
    def analyze_velocity(self):
        if 'product' not in self.column_mapping:
            return {}
        grp = self.df.groupby(self.column_mapping['product'])[self.column_mapping['quantity']].sum()
        days = 30
        if 'date' in self.column_mapping:
            days = max(1, (self.df[self.column_mapping['date']].max() - self.df[self.column_mapping['date']].min()).days)
        velocity = (grp / (days / 7)).round(1).to_dict()
        return {k: max(v, 0.1) for k, v in velocity.items()}
    
    def generate_recommendations(self, current_inventory_dict):
        recs = []
        velocity = self.analyze_velocity()
        for prod, vel in velocity.items():
            stock = current_inventory_dict.get(prod, 0)
            # Reorder
            if stock < vel * ML_CONFIG["low_stock_threshold_weeks"]:
                qty = int(vel * ML_CONFIG["reorder_multiplier"] * 2)
                recs.append({
                    "id": f"reorder_{int(time.time())}",
                    "type": "REORDER",
                    "product": prod,
                    "current_stock": stock,
                    "weekly_velocity": round(vel, 1),
                    "recommended_quantity": qty,
                    "confidence": 90,
                    "status": "pending"
                })
            # Promotion
            if stock > vel * ML_CONFIG["overstock_threshold_weeks"]:
                discount = "40%" if stock > vel * ML_CONFIG["severe_overstock_weeks"] else "30%"
                recs.append({
                    "id": f"promo_{int(time.time())}",
                    "type": "PROMOTION",
                    "product": prod,
                    "current_stock": stock,
                    "weekly_velocity": round(vel, 1),
                    "excess_weeks": round(stock / vel, 1),
                    "recommended_action": f"{discount} off flash sale",
                    "confidence": 92,
                    "status": "pending"
                })
        return recs
    
    def get_dataframe(self):
        return self.df

# ==================== MAIN UI ====================
st.title("Data Sources")

st.markdown("---")
st.subheader("1. Upload Sales History (Required)")
sales_file = st.file_uploader("Sales CSV (date, product, quantity)", type="csv", key="sales")

st.subheader("2. Upload Current Inventory (Makes Promotion Agent Work!)")
inventory_file = st.file_uploader("Stock CSV (product, current_stock / in_stock)", type="csv", key="stock")

if sales_file and st.button("Process Data & Generate AI Recommendations", type="primary", use_container_width=True):
    with st.spinner("AI analyzing sales + stock..."):
        processor = CSVProcessor()
        ok, msg = processor.load_csv(sales_file)
        if not ok:
            st.error(msg)
            st.stop()
        
        current_stock = load_inventory()
        if inventory_file:
            try:
                df_stock = pd.read_csv(inventory_file)
                df_stock.columns = [c.strip().lower().replace(' ', '_') for c in df_stock.columns]
                stock_col = next((c for c in ['current_stock','in_stock','stock','on_hand','quantity'] if c in df_stock.columns), None)
                if stock_col and 'product' in df_stock.columns:
                    for _, row in df_stock.iterrows():
                        p = str(row['product']).strip()
                        try:
                            current_stock[p] = int(float(row[stock_col]))
                        except:
                            pass
                    st.success(f"Loaded stock for {len([v for v in current_stock.values() if v>0])} products")
            except Exception as e:
                st.warning(f"Could not read inventory CSV: {e}")
        
        recs = processor.generate_recommendations(current_stock)
        save_sales_data(processor.get_dataframe().to_dict('records'))
        save_inventory(current_stock)
        save_recommendations(recs)
        
        st.success(f"AI Generated {len(recs)} Recommendations!")
        st.balloons()

st.markdown("---")
st.subheader("Shelf Scan")
col1, col2 = st.columns(2)
image_bytes = None
with col1:
    up = st.file_uploader("Upload photo", type=["png","jpg","jpeg"], key="up")
    if up:
        img = Image.open(up).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        st.image(image_bytes, use_column_width=True)
with col2:
    cam = st.camera_input("Take photo")
    if cam:
        image_bytes = cam.getvalue()
        st.image(image_bytes, use_column_width=True)

if image_bytes and st.button("Analyze Shelf", type="primary"):
    scanner = ShelfScanner()
    result = scanner.scan_shelf(image_bytes)
    if result['success']:
        inv = load_inventory()
        inv.update({p: {"quantity": q, "last_scanned": datetime.now().isoformat()} for p, q in result['products'].items()})
        save_inventory(inv)
        st.success("Inventory updated from photo!")
    else:
        st.error(result.get('error'))

# Developer panel
with st.sidebar:
    if st.checkbox("Show Config"):
        st.json({"GPT4_CONFIG": GPT4_CONFIG, "ML_CONFIG": ML_CONFIG})