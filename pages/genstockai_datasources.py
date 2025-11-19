import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, time as dt_time
import cv2
from PIL import Image
import io
import time

# ==================== DEVELOPER CONFIGURATION ====================
# Fully controllable — override in secrets.toml anytime
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

# ==================== YOUR SHELF SCANNER (unchanged) ====================
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
            return [(bbox, text, conf) for bbox, text, conf in results if conf >= self.confidence_threshold]
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
        estimated = {}
        for product, count in products.items():
            base = count
            for _, text, _ in detected_texts:
                numbers = [int(s) for s in text.split() if s.isdigit() and 1 <= int(s) <= 100]
                if numbers:
                    base = max(base, max(numbers))
            if product in ['red bull', 'pepsi', 'coca-cola']:
                estimated[product] = max(base * 3, 5)
            elif product in ['croissant', 'bagel', 'muffin']:
                estimated[product] = max(base * 2, 3)
            else:
                estimated[product] = max(base, 1)
        return estimated
    
    def scan_shelf(self, image_bytes):
        try:
            preprocessed = self._preprocess_image(image_bytes)
            texts = self._extract_text_with_ocr(preprocessed)
            if not texts:
                return {'success': False, 'error': 'No text detected', 'confidence': 0.0}
            matched = self._match_products(texts)
            if not matched:
                return {'success': False, 'error': 'No known products found', 'confidence': 0.3}
            quantities = self._estimate_quantities(matched, texts)
            conf = np.mean([c for _, _, c in texts])
            overall_conf = (conf + min(len(quantities)/5, 1.0)) / 2
            return {'success': True, 'products': quantities, 'confidence': float(overall_conf), 'detections_count': len(texts)}
        except Exception as e:
            return {'success': False, 'error': str(e), 'confidence': 0.0}

# ==================== FIXED & UPGRADED CSV PROCESSOR ====================
class CSVProcessor:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
    
    def load_csv(self, file):
        try:
            for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(file, encoding=enc)
                    break
                except:
                    continue
            if self.df is None:
                return False, "Could not read CSV"
            self.df.columns = [c.strip().lower().replace(' ', '_') for c in self.df.columns]
            self._detect_columns()
            return True, "CSV loaded"
        except Exception as e:
            return False, str(e)
    
    def _detect_columns(self):
        cols = self.df.columns.tolist()
        # Date
        for kw in ['date', 'time', 'transaction']:
            for c in cols:
                if kw in c:
                    self.column_mapping['date'] = c
                    self.df[c] = pd.to_datetime(self.df[c], errors='coerce')
                    break
        # Product
        for kw in ['product', 'item', 'name', 'description', 'sku']:
            for c in cols:
                if kw in c:
                    self.column_mapping['product'] = c
                    break
        # Quantity
        for kw in ['quantity', 'qty', 'units', 'sold']:
            for c in cols:
                if kw in c and 'price' not in c and 'stock' not in c:
                    self.column_mapping['quantity'] = c
                    break
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
        # Stock
        for kw in ['in_stock', 'current_stock', 'stock', 'inventory', 'on_hand']:
            for c in cols:
                if kw in c:
                    self.column_mapping['stock'] = c
                    break
    
    def analyze_product_performance(self):
        if not self.df.columns.intersection(['product', 'quantity']).any():
            return []
        grp = self.df.groupby(self.column_mapping['product'])[self.column_mapping['quantity']].agg(['sum', 'count'])
        result = grp.reset_index().rename(columns={'sum': 'total_quantity', 'count': 'transactions'})
        result['weekly_velocity'] = result['total_quantity'] / 4
        if 'date' in self.column_mapping:
            days = (self.df[self.column_mapping['date']].max() - self.df[self.column_mapping['date']].min()).days
            weeks = max(days / 7, 1)
            result['weekly_velocity'] = result['total_quantity'] / weeks
        return result.to_dict('records')
    
    def generate_recommendations(self, inventory=None):
        recommendations = []
        perf = self.analyze_product_performance()
        has_stock = 'stock' in self.column_mapping
        stock_col = self.column_mapping.get('stock')
        
        for row in perf:
            name = row['product']
            velocity = max(row['weekly_velocity'], 0.1)
            
            # Get current stock
            current_stock = 0
            if has_stock:
                prod_rows = self.df[self.df[self.column_mapping['product']] == name]
                if not prod_rows.empty and pd.notna(prod_rows[stock_col].iloc[-1]):
                    current_stock = int(float(prod_rows[stock_col].iloc[-1]))
            if current_stock == 0 and inventory and name in inventory:
                current_stock = inventory[name].get('quantity', 0)
            
            # Reorder
            if current_stock < velocity * ML_CONFIG["low_stock_threshold_weeks"]:
                qty = int(velocity * ML_CONFIG["reorder_multiplier"] * 2)
                recommendations.append({
                    'id': f"reorder_{int(time.time())}",
                    'type': 'REORDER',
                    'product': name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(velocity, 1),
                    'recommended_quantity': qty,
                    'confidence': 90,
                    'ai_agent': 'Reorder Agent',
                    'status': 'pending'
                })
            
            # Promotion
            if current_stock > velocity * ML_CONFIG["overstock_threshold_weeks"]:
                discount = "40%" if current_stock > velocity * ML_CONFIG["severe_overstock_weeks"] else "30%"
                recommendations.append({
                    'id': f"promo_{int(time.time())}",
                    'type': 'PROMOTION',
                    'product': name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(velocity, 1),
                    'excess_weeks': round(current_stock / velocity, 1),
                    'recommended_action': f"{discount} off flash sale",
                    'confidence': 92,
                    'ai_agent': 'Promotion Agent',
                    'status': 'pending'
                })
        
        return recommendations
    
    def get_dataframe(self):
        return self.df

# ==================== DATA PERSISTENCE ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")

def load_inventory():
    return json.load(open(INVENTORY_FILE, 'r')) if os.path.exists(INVENTORY_FILE) else {}

def save_inventory(inv):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inv, f, indent=2)

def save_recommendations(recs):
    with open(RECOMMENDATIONS_FILE, 'w') as f:
        json.dump(recs, f, indent=2, default=str)

# ==================== UI ====================
st.title("Data Sources")

st.markdown("---")
st.subheader("CSV Upload")
uploaded_file = st.file_uploader("Upload sales CSV", type="csv")
if uploaded_file and st.button("Process CSV", type="primary"):
    processor = CSVProcessor()
    ok, msg = processor.load_csv(uploaded_file)
    if ok:
        recs = processor.generate_recommendations(load_inventory())
        save_recommendations(recs)
        st.success(f"Generated {len(recs)} recommendations!")
    else:
        st.error(msg)

st.markdown("---")
st.subheader("Shelf Scan")
col1, col2 = st.columns(2)
image_bytes = None
with col1:
    up = st.file_uploader("Upload shelf photo", type=["png","jpg","jpeg"], key="up")
    if up:
        img = Image.open(up).convert('RGB')
        img.thumbnail((1200,1600))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        st.image(image_bytes, use_column_width=True)
with col2:
    cam = st.camera_input("Or take photo")
    if cam:
        image_bytes = cam.getvalue()
        st.image(image_bytes, use_column_width=True)

if image_bytes and st.button("Analyze Shelf", type="primary"):
    scanner = ShelfScanner()
    with st.spinner("Scanning..."):
        result = scanner.scan_shelf(image_bytes)
    if result['success']:
        inv = load_inventory()
        for p, q in result['products'].items():
            inv[p] = {"quantity": q, "last_scanned": datetime.now().isoformat()}
        save_inventory(inv)
        st.success(f"Found {len(result['products'])} products – inventory updated!")
    else:
        st.error(result.get('error', 'Failed'))

# Developer panel
with st.sidebar:
    if st.checkbox("Show Config"):
        st.json({"GPT4_CONFIG": GPT4_CONFIG, "ML_CONFIG": ML_CONFIG}, expanded=False)