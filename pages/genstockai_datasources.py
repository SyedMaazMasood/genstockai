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

# ==================== AI PROVIDER CONFIGURATION ====================
"""
DUAL AI SYSTEM: Switch between Groq (FREE) and OpenAI (PAID)

To switch providers, update secrets.toml:
    AI_PROVIDER = "groq"     # FREE - Recommended for development
    AI_PROVIDER = "openai"   # PAID - Use when you have credits

Groq: Free, fast Llama 3.1 (70B) - comparable to GPT-4
OpenAI: Paid GPT-4o-mini - slightly better quality but costs money
"""

AI_PROVIDER = st.secrets.get("AI_PROVIDER", "groq").lower()
AI_ENABLED = False
ai_client = None

# Try to initialize the selected AI provider
if AI_PROVIDER == "groq":
    try:
        from groq import Groq
        ai_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        AI_ENABLED = True
        AI_MODEL = "llama-3.3-70b-versatile"  # Groq's best free model
        st.sidebar.success(f"✅ AI Active: Groq (FREE)")
    except Exception as e:
        st.sidebar.error(f"❌ Groq Error: {e}")
        st.sidebar.info("💡 Add GROQ_API_KEY to secrets.toml")

elif AI_PROVIDER == "openai":
    try:
        from openai import OpenAI
        ai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        AI_ENABLED = True
        AI_MODEL = "gpt-4o-mini"  # OpenAI's cheapest GPT-4 model
        st.sidebar.success(f"✅ AI Active: OpenAI GPT-4o-mini (PAID)")
        st.sidebar.warning("⚠️ Using paid API - costs apply")
    except Exception as e:
        st.sidebar.error(f"❌ OpenAI Error: {e}")
        st.sidebar.info("💡 Switch to Groq (free) in secrets.toml")

else:
    st.sidebar.error(f"❌ Invalid AI_PROVIDER: {AI_PROVIDER}")
    st.sidebar.info("Set AI_PROVIDER to 'groq' or 'openai'")

# ML Configuration
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
def call_ai_for_reorder_analysis(product, velocity, current_stock, sales_history):
    """
    Uses REAL AI (Groq or OpenAI) to analyze reorder needs
    
    GROQ (FREE): Uses Llama 3.1 70B - very fast and free
    OPENAI (PAID): Uses GPT-4o-mini - costs ~$0.0003 per call
    
    To switch: Change AI_PROVIDER in secrets.toml
    """
    if not AI_ENABLED:
        # Fallback: Rule-based logic (no AI)
        weeks_supply = current_stock / velocity if velocity > 0 else 999
        if weeks_supply < ML_CONFIG["low_stock_threshold_weeks"]:
            return {
                "should_reorder": True,
                "quantity": int(velocity * ML_CONFIG["reorder_multiplier"] * 2),
                "reason": f"Low stock: {weeks_supply:.1f} weeks supply",
                "confidence": 88
            }
        return {"should_reorder": False}
    
    try:
        prompt = f"""You are an inventory management AI. Analyze this product and decide if we should reorder.

Product: {product}
Current Stock: {current_stock} units
Weekly Sales: {velocity} units/week
Trend: {sales_history}

Respond ONLY with valid JSON (no markdown):
{{
    "should_reorder": true or false,
    "quantity": integer (if reordering),
    "reason": "brief explanation",
    "confidence": integer 0-100
}}

Consider lead time (2-3 days) and safety stock. Recommend reorder if stock will run out in less than 2 weeks."""

        # Call Groq or OpenAI (same API format)
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are an inventory analyst. Respond only with JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        return result
        
    except Exception as e:
        st.warning(f"AI Error: {e}. Using fallback logic.")
        weeks_supply = current_stock / velocity if velocity > 0 else 999
        if weeks_supply < 1.5:
            return {
                "should_reorder": True,
                "quantity": int(velocity * 2 * 2),
                "reason": f"Low stock: {weeks_supply:.1f} weeks",
                "confidence": 85
            }
        return {"should_reorder": False}

def call_ai_for_promotion_strategy(product, velocity, current_stock, excess_weeks):
    """
    Uses REAL AI (Groq or OpenAI) to create promotion strategies
    
    GROQ (FREE): Creative and fast
    OPENAI (PAID): Slightly more sophisticated responses
    """
    if not AI_ENABLED:
        discount = "40%" if excess_weeks > 12 else "30%"
        return {
            "create_promotion": True,
            "strategy": f"{discount} off flash sale",
            "reason": f"Overstock: {excess_weeks:.1f} weeks",
            "confidence": 90
        }
    
    try:
        prompt = f"""You are a retail promotion expert. Create a promotion for this overstocked item.

Product: {product}
Current Stock: {current_stock} units
Weekly Sales: {velocity} units/week
Weeks Supply: {excess_weeks:.1f} weeks (OVERSTOCKED!)

Respond ONLY with JSON (no markdown):
{{
    "create_promotion": true,
    "strategy": "creative promotion idea",
    "discount_percentage": integer 10-50,
    "reason": "why this will work",
    "confidence": integer 0-100
}}

Make it attractive to customers without excessive loss."""

        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            temperature=0.5,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a promotion expert. Respond only with JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        return result
        
    except Exception as e:
        st.warning(f"AI Error: {e}. Using fallback.")
        discount = "40%" if excess_weeks > 12 else "30%"
        return {
            "create_promotion": True,
            "strategy": f"{discount} off flash sale",
            "reason": f"Overstock: {excess_weeks:.1f} weeks",
            "confidence": 85
        }

# ==================== SHELF SCANNER ====================
class ShelfScanner:
    """
    Computer Vision shelf scanner using YOLOv8 + EasyOCR
    This is REAL AI that runs locally (no API costs)
    """
    def __init__(self):
        self.yolo_model = None
        self.ocr_reader = None

    def _load_yolo(self):
        if hasattr(self, 'yolo_loaded'):
            return self.yolo_model
            
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_loaded = True
            return self.yolo_model
        except Exception as e:
            st.warning(f"YOLO not available: {e}")
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
                            
                            if 'bottle' in label or 'cup' in label or 'bowl' in label:
                                detected[label] = detected.get(label, 0) + 1
                    
                    if detected:
                        st.info(f"🤖 YOLO detected: {detected}")
                    
                except Exception as e:
                    st.warning(f"YOLO error: {e}")

            # Fallback to OCR
            if not detected:
                ocr = self._load_ocr()
                if ocr and ocr != False:
                    try:
                        ocr_results = ocr.readtext(img, detail=0)
                        all_text = ' '.join([t.lower() for t in ocr_results])
                        
                        if all_text:
                            st.info(f"📝 OCR detected: {all_text[:100]}")
                        
                        products = ['milk', 'water', 'juice', 'soda', 'coffee', 'tea', 
                                   'chips', 'cookies', 'candy', 'bread', 'cheese']
                        for product in products:
                            if product in all_text:
                                detected[product] = max(all_text.count(product) * 3, 5)
                    except Exception as e:
                        st.warning(f"OCR error: {e}")

            if not detected:
                return {'success': False, 'error': 'No products detected. Try better lighting.'}

            quantities = {}
            for product, count in detected.items():
                quantities[product] = max(count, 5)

            return {
                'success': True,
                'products': quantities,
                'confidence': 0.85,
                'method': 'YOLOv8 + OCR (Local AI)'
            }

        except Exception as e:
            return {'success': False, 'error': f'Scan failed: {str(e)}'}

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
            return True, "Loaded successfully"
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
        """
        Generate recommendations using REAL AI
        
        Current AI: {AI_PROVIDER.upper()} ({AI_MODEL})
        Cost: {'FREE' if AI_PROVIDER == 'groq' else 'PAID'}
        
        To switch providers, update secrets.toml
        """
        recs = []
        velocity = self.analyze_velocity()
        
        if not velocity:
            st.warning("No products found in sales data!")
            return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_products = len(velocity)
        
        for idx, (prod, vel) in enumerate(velocity.items()):
            status_text.text(f"🤖 AI analyzing: {prod}... ({idx+1}/{total_products})")
            progress_bar.progress((idx + 1) / total_products)
            
            stock = current_inventory_dict.get(prod, 0)
            weeks_supply = stock / vel if vel > 0 else 999
            
            # Get trend
            if 'date' in self.column_mapping and 'product' in self.column_mapping:
                product_df = self.df[self.df[self.column_mapping['product']] == prod]
                if len(product_df) >= 4:
                    mid = len(product_df) // 2
                    first = product_df.iloc[:mid][self.column_mapping['quantity']].sum()
                    second = product_df.iloc[mid:][self.column_mapping['quantity']].sum()
                    trend = "growing" if second > first * 1.2 else "declining" if second < first * 0.8 else "stable"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
            
            # REORDER CHECK
            if weeks_supply < ML_CONFIG["low_stock_threshold_weeks"]:
                ai_result = call_ai_for_reorder_analysis(prod, vel, stock, trend)
                
                if ai_result.get("should_reorder"):
                    agent_name = f"Reorder Agent ({AI_PROVIDER.upper()})" if AI_ENABLED else "Reorder Agent (Rule-based)"
                    
                    recs.append({
                        "id": f"reorder_{prod.replace(' ', '_')}_{int(time.time())}_{idx}",
                        "type": "REORDER",
                        "product": prod,
                        "current_stock": stock,
                        "weekly_velocity": round(vel, 1),
                        "recommended_quantity": ai_result.get("quantity", int(vel * 4)),
                        "reason": ai_result.get("reason", "Low stock detected"),
                        "confidence": ai_result.get("confidence", 90),
                        "ai_agent": agent_name,
                        "status": "pending"
                    })
            
            # PROMOTION CHECK
            elif weeks_supply > ML_CONFIG["overstock_threshold_weeks"]:
                ai_result = call_ai_for_promotion_strategy(prod, vel, stock, weeks_supply)
                
                if ai_result.get("create_promotion"):
                    agent_name = f"Promotion Agent ({AI_PROVIDER.upper()})" if AI_ENABLED else "Promotion Agent (Rule-based)"
                    
                    recs.append({
                        "id": f"promo_{prod.replace(' ', '_')}_{int(time.time())}_{idx}",
                        "type": "PROMOTION",
                        "product": prod,
                        "current_stock": stock,
                        "weekly_velocity": round(vel, 1),
                        "excess_weeks": round(weeks_supply, 1),
                        "recommended_action": ai_result.get("strategy", "Discount sale"),
                        "reason": ai_result.get("reason", f"Overstock: {weeks_supply:.1f} weeks"),
                        "confidence": ai_result.get("confidence", 92),
                        "ai_agent": agent_name,
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

# Show AI status banner
if AI_ENABLED:
    if AI_PROVIDER == "groq":
        st.success(f"✅ AI Active: Groq Llama 3.1 (70B) - **FREE & UNLIMITED**")
        st.info("💡 To use OpenAI GPT-4 (paid), change AI_PROVIDER to 'openai' in secrets.toml")
    else:
        st.success(f"✅ AI Active: OpenAI {AI_MODEL}")
        st.warning("⚠️ **Using PAID API** - Each analysis costs ~$0.01. Switch to Groq for FREE!")
else:
    st.error("❌ No AI configured. Add GROQ_API_KEY or OPENAI_API_KEY to secrets.toml")

st.markdown("---")

# ==================== CSV UPLOAD ====================
st.subheader("1️⃣ Upload Sales History")
st.markdown("Upload CSV with: date, product, quantity")
sales_file = st.file_uploader("Sales CSV", type="csv", key="sales")

st.subheader("2️⃣ Upload Current Inventory (Optional)")
st.markdown("Upload CSV with: product, current_stock")
inventory_file = st.file_uploader("Inventory CSV", type="csv", key="stock")

if sales_file:
    if st.button("🤖 Process with AI", type="primary", use_container_width=True):
        with st.spinner(f"🤖 AI analyzing with {AI_PROVIDER.upper()}..."):
            processor = CSVProcessor()
            ok, msg = processor.load_csv(sales_file)
            if not ok:
                st.error(f"❌ {msg}")
                st.stop()
            
            st.success(f"✅ Loaded {len(processor.df)} transactions")
            
            current_stock = load_inventory()
            
            if inventory_file:
                try:
                    df_stock = pd.read_csv(inventory_file)
                    df_stock.columns = [c.strip().lower().replace(' ', '_') for c in df_stock.columns]
                    
                    stock_col = next((c for c in ['current_stock','in_stock','stock','quantity'] if c in df_stock.columns), None)
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
                        st.success(f"✅ Loaded {len(current_stock)} products")
                except Exception as e:
                    st.warning(f"⚠️ Inventory error: {e}")
            
            inv_qty = {}
            for p, d in current_stock.items():
                inv_qty[p] = d.get('quantity', 0) if isinstance(d, dict) else d
            
            if AI_ENABLED:
                st.info(f"🤖 Calling {AI_PROVIDER.upper()} AI... (30-60s)")
            
            recs = processor.generate_recommendations_with_ai(inv_qty)
            
            save_sales_data(processor.get_dataframe().to_dict('records'))
            save_inventory(current_stock)
            save_recommendations(recs)
            
            st.success(f"✅ Generated {len(recs)} AI recommendations!")
            st.balloons()
            
            if recs:
                st.info("👉 Go to **Approval Queue** to review!")
            else:
                st.info("ℹ️ No recommendations needed!")

st.markdown("---")

# ==================== SHELF SCANNER ====================
st.subheader("3️⃣ 📸 Shelf Scanner (Computer Vision)")
st.markdown("Uses YOLOv8 + OCR to detect products - **Runs locally, no API costs**")

col1, col2 = st.columns(2)
image_bytes = None

with col1:
    st.markdown("**Upload Photo**")
    up = st.file_uploader("", type=["png","jpg","jpeg"], key="up")
    if up:
        img = Image.open(up).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        st.image(image_bytes, use_container_width=True)

with col2:
    st.markdown("**Take Photo**")
    cam = st.camera_input("", label_visibility="collapsed")
    if cam:
        image_bytes = cam.getvalue()
        st.image(image_bytes, use_container_width=True)

if image_bytes:
    if st.button("🤖 Scan Shelf", type="primary", use_container_width=True):
        with st.spinner("🤖 Scanning with computer vision..."):
            scanner = ShelfScanner()
            result = scanner.scan_shelf(image_bytes)
            
            if result['success']:
                st.success(f"✅ Detected {len(result['products'])} products!")
                
                inv = load_inventory()
                for prod, qty in result['products'].items():
                    inv[prod] = {
                        "quantity": qty,
                        "last_scanned": datetime.now().isoformat(),
                        "source": "shelf_scan",
                        "confidence": result.get('confidence', 0.85)
                    }
                
                if save_inventory(inv):
                    st.success("✅ Inventory updated!")
                    
                    st.markdown("### 📦 Detected:")
                    for prod, qty in result['products'].items():
                        st.metric(prod.title(), f"{qty} units")
                    
                    if st.button("📊 View in Analytics"):
                        st.switch_page("pages/genstockai_analytics.py")
            else:
                st.error(f"❌ {result.get('error')}")

st.markdown("---")

# Current inventory
st.subheader("📦 Current Inventory")
inv = load_inventory()

if inv:
    st.markdown(f"**{len(inv)} products tracked**")
    
    data = []
    for p, d in inv.items():
        if isinstance(d, dict):
            data.append({
                'Product': p,
                'Qty': d.get('quantity', 0),
                'Updated': d.get('last_scanned', d.get('last_updated', 'N/A'))[:10],
                'Source': d.get('source', 'manual')
            })
        else:
            data.append({'Product': p, 'Qty': d, 'Updated': 'N/A', 'Source': 'manual'})
    
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
else:
    st.info("No inventory yet. Scan a shelf or upload CSV!")

# Developer tools
with st.sidebar:
    st.markdown("### 🔧 Developer Tools")
    
    st.markdown("**AI Provider:**")
    st.code(f"{AI_PROVIDER.upper()} - {AI_MODEL if AI_ENABLED else 'Not configured'}")
    
    if st.checkbox("Show Full Config"):
        st.json({
            "AI_Provider": AI_PROVIDER,
            "AI_Enabled": AI_ENABLED,
            "Model": AI_MODEL if AI_ENABLED else None,
            "Cost": "FREE" if AI_PROVIDER == "groq" else "PAID",
            "ML_Config": ML_CONFIG
        })
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All Data"):
        for f in [SALES_DATA_FILE, INVENTORY_FILE, RECOMMENDATIONS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.success("Cleared!")
        st.rerun()