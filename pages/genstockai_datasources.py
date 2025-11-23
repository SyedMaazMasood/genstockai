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
from ultralytics import YOLO
import torch
from openai import OpenAI

# ==================== AI CONFIGURATION ====================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    AI_ENABLED = True
except:
    st.warning("⚠️ OpenAI API key not found. AI features will use fallback logic.")
    AI_ENABLED = False

GPT4_CONFIG = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.3,
    "max_tokens": 500,
}

ML_CONFIG = {
    "growth_threshold": 1.2,
    "decline_threshold": 0.8,
    "min_data_points": 4,
    "base_confidence": 88,
    "reorder_multiplier": 2.0,
    "low_stock_threshold_weeks": 1.5,
    "overstock_threshold_weeks": 6.0,
    "severe_overstock_weeks": 12.0,
}

# ==================== DATA PERSISTENCE ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")

def save_sales_data(sales_records):
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
    try:
        with open(INVENTORY_FILE, 'w') as f:
            json.dump(inventory, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving inventory: {e}")
        return False

def save_recommendations(recommendations):
    try:
        with open(RECOMMENDATIONS_FILE, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving recommendations: {e}")

# ==================== REAL AI FUNCTIONS ====================
def call_gpt4_for_reorder_analysis(product, velocity, current_stock, sales_history):
    """Uses real GPT-4 to analyze reorder needs"""
    if not AI_ENABLED:
        # Fallback logic
        weeks_supply = current_stock / velocity if velocity > 0 else 999
        if weeks_supply < ML_CONFIG["low_stock_threshold_weeks"]:
            return {
                "should_reorder": True,
                "quantity": int(velocity * ML_CONFIG["reorder_multiplier"] * 2),
                "reason": f"Low stock detected: only {weeks_supply:.1f} weeks supply remaining",
                "confidence": 88
            }
        return {"should_reorder": False}
    
    try:
        prompt = f"""You are an inventory management expert. Analyze this product data and provide reorder recommendation.

Product: {product}
Current Stock: {current_stock} units
Weekly Sales Velocity: {velocity} units/week
Recent Sales Trend: {sales_history if sales_history else 'No trend data'}

Provide your analysis in JSON format:
{{
    "should_reorder": true/false,
    "quantity": recommended_order_quantity (integer),
    "reason": "brief explanation",
    "confidence": confidence_score_0_to_100 (integer)
}}

Consider:
- Lead time of 2-3 days
- Safety stock buffer
- Sales trends
- Avoid stockouts"""

        response = client.chat.completions.create(
            model=GPT4_CONFIG["model"],
            temperature=GPT4_CONFIG["temperature"],
            max_tokens=GPT4_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": "You are an AI inventory analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        st.warning(f"GPT-4 call failed: {e}. Using fallback logic.")
        # Fallback
        weeks_supply = current_stock / velocity if velocity > 0 else 999
        if weeks_supply < 1.5:
            return {
                "should_reorder": True,
                "quantity": int(velocity * 2 * 2),
                "reason": f"Low stock: {weeks_supply:.1f} weeks supply",
                "confidence": 85
            }
        return {"should_reorder": False}

def call_gpt4_for_promotion_strategy(product, velocity, current_stock, excess_weeks):
    """Uses real GPT-4 to create promotion strategies"""
    if not AI_ENABLED:
        discount = "40%" if excess_weeks > 12 else "30%"
        return {
            "create_promotion": True,
            "strategy": f"{discount} off flash sale",
            "reason": f"Overstock: {excess_weeks:.1f} weeks of inventory",
            "confidence": 90
        }
    
    try:
        prompt = f"""You are a retail promotion strategist. Create a promotion strategy for this overstocked item.

Product: {product}
Current Stock: {current_stock} units
Weekly Sales Velocity: {velocity} units/week
Weeks of Supply: {excess_weeks:.1f} weeks (OVERSTOCKED)

Provide strategy in JSON format:
{{
    "create_promotion": true,
    "strategy": "specific promotion idea",
    "discount_percentage": integer (10-50),
    "reason": "brief explanation",
    "confidence": confidence_score_0_to_100 (integer)
}}

Create an attractive promotion that will move inventory quickly without excessive loss."""

        response = client.chat.completions.create(
            model=GPT4_CONFIG["model"],
            temperature=0.5,  # Slightly higher for creative promotions
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a creative retail promotion expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        st.warning(f"GPT-4 call failed for promotion: {e}")
        discount = "40%" if excess_weeks > 12 else "30%"
        return {
            "create_promotion": True,
            "strategy": f"{discount} off flash sale",
            "reason": f"Overstock: {excess_weeks:.1f} weeks",
            "confidence": 85
        }

# ==================== SHELF SCANNER ====================
class ShelfScanner:
    def __init__(self):
        self.yolo_model = None
        self.ocr_reader = None
        self.product_map = {
            'bottle': 'beverage',
            'cup': 'cup',
            'person': None,  # Ignore
            'chair': None,
            'cell phone': None,
            # Add more YOLO class mappings as needed
        }

    def _load_yolo(self):
        if hasattr(self, 'yolo_loaded'):
            return self.yolo_model
            
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_loaded = True
            return self.yolo_model
        except Exception as e:
            st.warning(f"YOLO model not available: {e}")
            self.yolo_model = None
            self.yolo_loaded = True
            return None

    def _load_ocr(self):
        if self.ocr_reader is None:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except:
                self.ocr_reader = False
        return self.ocr_reader

    def scan_shelf(self, image_bytes):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {'success': False, 'error': 'Invalid image'}

            img = cv2.resize(img, (640, 640))
            detected = {}

            # Try YOLO detection
            model = self._load_yolo()
            if model:
                try:
                    results = model(img, conf=0.3, verbose=False, device='cpu')
                    for r in results:
                        for box in r.boxes:
                            label = r.names[int(box.cls[0])].lower()
                            conf = float(box.conf[0])
                            
                            # Map YOLO classes to products
                            if 'bottle' in label or 'cup' in label:
                                product_name = label
                                detected[product_name] = detected.get(product_name, 0) + 1
                    
                    st.info(f"🤖 YOLO detected: {detected}")
                    
                except Exception as e:
                    st.warning(f"YOLO processing error: {e}")

            # Fallback to OCR
            if not detected:
                ocr = self._load_ocr()
                if ocr and ocr != False:
                    try:
                        ocr_results = ocr.readtext(img, detail=0)
                        all_text = ' '.join([t.lower() for t in ocr_results])
                        st.info(f"📝 OCR detected text: {all_text[:100]}...")
                        
                        # Simple product detection from text
                        common_products = ['milk', 'water', 'juice', 'soda', 'coffee', 'tea']
                        for product in common_products:
                            if product in all_text:
                                detected[product] = all_text.count(product) * 3
                    except Exception as e:
                        st.warning(f"OCR error: {e}")

            if not detected:
                return {'success': False, 'error': 'No products recognized. Try a clearer photo or manually enter inventory.'}

            # Convert counts to reasonable quantities
            quantities = {}
            for product, count in detected.items():
                quantities[product] = max(count, 5)  # Minimum 5 units per detected item

            return {
                'success': True,
                'products': quantities,
                'confidence': 0.85,
                'method': 'YOLOv8 + OCR'
            }

        except Exception as e:
            return {'success': False, 'error': f'Processing failed: {str(e)}'}

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
                    break
        for kw in ['product', 'item', 'name', 'description', 'sku']:
            for c in cols:
                if kw in c:
                    self.column_mapping['product'] = c
                    break
        for kw in ['quantity', 'qty', 'units', 'sold']:
            for c in cols:
                if kw in c and 'price' not in c:
                    self.column_mapping['quantity'] = c
                    break
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
    
    def analyze_velocity(self):
        if 'product' not in self.column_mapping:
            return {}
        grp = self.df.groupby(self.column_mapping['product'])[self.column_mapping['quantity']].sum()
        days = 30
        if 'date' in self.column_mapping:
            date_range = (self.df[self.column_mapping['date']].max() - self.df[self.column_mapping['date']].min()).days
            days = max(date_range, 1)
        velocity = (grp / (days / 7)).round(1).to_dict()
        return {k: max(v, 0.1) for k, v in velocity.items()}
    
    def generate_recommendations_with_ai(self, current_inventory_dict):
        """Generate recommendations using REAL GPT-4"""
        recs = []
        velocity = self.analyze_velocity()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_products = len(velocity)
        
        for idx, (prod, vel) in enumerate(velocity.items()):
            status_text.text(f"🤖 AI analyzing {prod}... ({idx+1}/{total_products})")
            progress_bar.progress((idx + 1) / total_products)
            
            stock = current_inventory_dict.get(prod, 0)
            weeks_supply = stock / vel if vel > 0 else 999
            
            # Get recent sales trend
            if 'date' in self.column_mapping and 'product' in self.column_mapping:
                product_df = self.df[self.df[self.column_mapping['product']] == prod]
                if len(product_df) >= 4:
                    mid = len(product_df) // 2
                    first_half = product_df.iloc[:mid][self.column_mapping['quantity']].sum()
                    second_half = product_df.iloc[mid:][self.column_mapping['quantity']].sum()
                    trend = "growing" if second_half > first_half * 1.2 else "stable"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
            
            # REORDER CHECK with AI
            if weeks_supply < ML_CONFIG["low_stock_threshold_weeks"]:
                ai_result = call_gpt4_for_reorder_analysis(prod, vel, stock, trend)
                
                if ai_result.get("should_reorder"):
                    recs.append({
                        "id": f"reorder_{prod}_{int(time.time())}_{idx}",
                        "type": "REORDER",
                        "product": prod,
                        "current_stock": stock,
                        "weekly_velocity": round(vel, 1),
                        "recommended_quantity": ai_result.get("quantity", int(vel * 2 * 2)),
                        "reason": ai_result.get("reason", "Low stock detected"),
                        "confidence": ai_result.get("confidence", 90),
                        "ai_agent": "Reorder Agent (GPT-4)" if AI_ENABLED else "Reorder Agent (Rule-based)",
                        "status": "pending"
                    })
            
            # PROMOTION CHECK with AI
            elif weeks_supply > ML_CONFIG["overstock_threshold_weeks"]:
                ai_result = call_gpt4_for_promotion_strategy(prod, vel, stock, weeks_supply)
                
                if ai_result.get("create_promotion"):
                    recs.append({
                        "id": f"promo_{prod}_{int(time.time())}_{idx}",
                        "type": "PROMOTION",
                        "product": prod,
                        "current_stock": stock,
                        "weekly_velocity": round(vel, 1),
                        "excess_weeks": round(weeks_supply, 1),
                        "recommended_action": ai_result.get("strategy", "Discount promotion"),
                        "reason": ai_result.get("reason", f"Overstock: {weeks_supply:.1f} weeks supply"),
                        "confidence": ai_result.get("confidence", 92),
                        "ai_agent": "Promotion Agent (GPT-4)" if AI_ENABLED else "Promotion Agent (Rule-based)",
                        "status": "pending"
                    })
        
        progress_bar.empty()
        status_text.empty()
        
        return recs
    
    def get_dataframe(self):
        return self.df

# ==================== MAIN UI ====================
st.title("📊 Data Sources")
st.markdown("Upload sales data and scan shelves to power AI recommendations")

# Show AI status
if AI_ENABLED:
    st.success("✅ AI Powered: OpenAI GPT-4 Active")
else:
    st.warning("⚠️ AI Fallback Mode: Using rule-based logic (add API key to enable GPT-4)")

st.markdown("---")

# ==================== CSV UPLOAD ====================
st.subheader("1️⃣ Upload Sales History (Required)")
sales_file = st.file_uploader("Sales CSV (date, product, quantity)", type="csv", key="sales")

st.subheader("2️⃣ Upload Current Inventory (Optional - Makes AI More Accurate)")
inventory_file = st.file_uploader("Stock CSV (product, current_stock)", type="csv", key="stock")

if sales_file and st.button("🤖 Process Data & Generate AI Recommendations", type="primary", use_container_width=True):
    with st.spinner("🤖 AI analyzing your data..."):
        processor = CSVProcessor()
        ok, msg = processor.load_csv(sales_file)
        if not ok:
            st.error(f"❌ {msg}")
            st.stop()
        
        st.success(f"✅ Loaded {len(processor.df)} sales transactions")
        
        # Load current inventory
        current_stock = load_inventory()
        
        # Process inventory CSV if provided
        if inventory_file:
            try:
                df_stock = pd.read_csv(inventory_file)
                df_stock.columns = [c.strip().lower().replace(' ', '_') for c in df_stock.columns]
                
                stock_col = next((c for c in ['current_stock','in_stock','stock','on_hand','quantity'] if c in df_stock.columns), None)
                product_col = next((c for c in ['product','item','name'] if c in df_stock.columns), None)
                
                if stock_col and product_col:
                    for _, row in df_stock.iterrows():
                        p = str(row[product_col]).strip()
                        try:
                            current_stock[p] = {
                                "quantity": int(float(row[stock_col])),
                                "last_updated": datetime.now().isoformat(),
                                "source": "csv_upload"
                            }
                        except:
                            pass
                    st.success(f"✅ Loaded inventory for {len(current_stock)} products")
                else:
                    st.warning("⚠️ Could not find 'product' and 'stock' columns in inventory CSV")
            except Exception as e:
                st.warning(f"⚠️ Could not read inventory CSV: {e}")
        
        # Convert inventory format for velocity calculations
        inventory_quantities = {}
        for prod, data in current_stock.items():
            if isinstance(data, dict):
                inventory_quantities[prod] = data.get('quantity', 0)
            else:
                inventory_quantities[prod] = data
        
        # Generate AI recommendations
        st.info("🤖 Calling GPT-4 for each product analysis... (this may take 30-60 seconds)")
        recs = processor.generate_recommendations_with_ai(inventory_quantities)
        
        # Save everything
        save_sales_data(processor.get_dataframe().to_dict('records'))
        save_inventory(current_stock)
        save_recommendations(recs)
        
        st.success(f"✅ AI Generated {len(recs)} Recommendations!")
        st.balloons()
        
        if recs:
            st.info("👉 Go to **Approval Queue** to review AI recommendations!")

st.markdown("---")

# ==================== SHELF SCANNER ====================
st.subheader("3️⃣ 📸 Shelf Scanner (Computer Vision)")
st.markdown("Take a photo of your shelf and AI will detect products and update inventory")

col1, col2 = st.columns(2)
image_bytes = None

with col1:
    st.markdown("**Upload Photo**")
    uploaded = st.file_uploader("Choose image", type=["png","jpg","jpeg"], key="upload", label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        st.image(image_bytes, use_container_width=True)

with col2:
    st.markdown("**Take Photo**")
    camera = st.camera_input("Camera", label_visibility="collapsed")
    if camera:
        image_bytes = camera.getvalue()
        st.image(image_bytes, use_container_width=True)

if image_bytes and st.button("🤖 Analyze Shelf with AI", type="primary", use_container_width=True):
    with st.spinner("🤖 AI scanning shelf..."):
        scanner = ShelfScanner()
        result = scanner.scan_shelf(image_bytes)
        
        if result['success']:
            st.success(f"✅ Detected {len(result['products'])} product types!")
            
            # Load existing inventory
            inv = load_inventory()
            
            # Update with detected quantities
            for product, quantity in result['products'].items():
                inv[product] = {
                    "quantity": quantity,
                    "last_scanned": datetime.now().isoformat(),
                    "source": "shelf_scan",
                    "confidence": result.get('confidence', 0.85)
                }
            
            # Save inventory
            if save_inventory(inv):
                st.success("✅ Inventory updated from photo!")
                
                # Show what was detected
                st.markdown("### 📦 Detected Inventory:")
                for product, quantity in result['products'].items():
                    st.metric(product.title(), f"{quantity} units")
                
                st.info("✨ Inventory has been updated and will now appear on the Analytics page!")
                
                # Force refresh by clearing cache
                if st.button("🔄 Go to Analytics to See Updated Inventory"):
                    st.switch_page("pages/genstockai_analytics.py")
            else:
                st.error("❌ Failed to save inventory")
        else:
            st.error(f"❌ {result.get('error', 'Unknown error')}")
            st.info("💡 Try:\n- Better lighting\n- Closer photo\n- Clearer product labels\n- Or manually enter inventory data")

st.markdown("---")

# Current inventory display
st.subheader("📦 Current Inventory")
current_inv = load_inventory()

if current_inv:
    st.markdown(f"**Tracking {len(current_inv)} products**")
    
    inv_data = []
    for product, data in current_inv.items():
        if isinstance(data, dict):
            qty = data.get('quantity', 0)
            last_update = data.get('last_scanned', data.get('last_updated', 'N/A'))
            source = data.get('source', 'manual')
        else:
            qty = data
            last_update = 'N/A'
            source = 'manual'
        
        inv_data.append({
            'Product': product,
            'Quantity': qty,
            'Last Updated': last_update[:10] if last_update != 'N/A' else 'N/A',
            'Source': source
        })
    
    df_inv = pd.DataFrame(inv_data)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)
else:
    st.info("No inventory data yet. Scan a shelf or upload inventory CSV to get started!")

# Developer panel
with st.sidebar:
    st.markdown("### 🔧 Developer Panel")
    
    if st.checkbox("Show Config"):
        st.json({
            "AI_ENABLED": AI_ENABLED,
            "GPT4_MODEL": GPT4_CONFIG["model"],
            "ML_CONFIG": ML_CONFIG
        })
    
    if st.button("🗑️ Clear All Data (Dev)"):
        for file in [SALES_DATA_FILE, INVENTORY_FILE, RECOMMENDATIONS_FILE]:
            if os.path.exists(file):
                os.remove(file)
        st.success("All data cleared!")
        st.rerun()