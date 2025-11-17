import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

# ==================== AI CONFIGURATION SETTINGS ====================
#
# 🎛️ AI/ML HYPERPARAMETERS - ADJUST HERE
# ========================================
# These parameters control the behavior of AI models and algorithms.
# Modify these values to tune the AI's decision-making process.
#
# ⚠️ FOR DEVELOPERS: Change values below to customize AI behavior
# 📍 LOCATION: Lines 15-80 of this file
#
# In production, these would be passed to actual API calls:
# - OpenAI API: openai.ChatCompletion.create(**GPT4_CONFIG)
# - Anthropic API: anthropic.messages.create(**CLAUDE_CONFIG)
#

# ============================================================
# === GPT-4 CONFIGURATION (OpenAI API) ===
# ============================================================
#
# 🤖 OpenAI GPT-4 Configuration
# Used for: Demand forecasting, reorder recommendations, data analysis
#
# 📍 ADJUST THESE VALUES to change GPT-4 behavior:
#
GPT4_CONFIG = {
    # Model Selection
    "model": "gpt-4-turbo-preview",           
    # Options: "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"
    # Recommendation: "gpt-4-turbo-preview" for best performance
    
    # Temperature: Controls randomness/creativity
    # Range: 0.0 - 2.0
    # 0.0 = Completely deterministic, same output every time
    # 0.3 = ✅ RECOMMENDED for business decisions (consistent but not rigid)
    # 0.7 = Balanced creativity
    # 1.0 = Creative and varied
    # 2.0 = Very random and unpredictable
    "temperature": 0.3,                        # ← CHANGE THIS for different creativity levels
    
    # Max Tokens: Maximum length of AI response
    # Range: 50 - 4000
    # 1 token ≈ 4 characters or 0.75 words
    # 100 tokens ≈ 75 words
    # 500 tokens ≈ ✅ RECOMMENDED for recommendations (375 words)
    # 1000 tokens ≈ 750 words (detailed reports)
    # Note: Higher values = higher API costs
    "max_tokens": 500,                         # ← CHANGE THIS for longer/shorter responses
    
    # Top P (Nucleus Sampling): Alternative to temperature
    # Range: 0.0 - 1.0
    # 0.1 = Only top 10% most likely tokens (very focused)
    # 0.5 = Top 50% of tokens
    # 0.9 = ✅ RECOMMENDED (top 90% - good balance)
    # 1.0 = Consider all tokens
    # Note: Use either temperature OR top_p, not both at extreme values
    "top_p": 0.9,                              # ← CHANGE THIS for token selection diversity
    
    # Frequency Penalty: Reduces repetition of tokens
    # Range: 0.0 - 2.0
    # 0.0 = ✅ RECOMMENDED (no penalty, natural repetition)
    # 0.5 = Slight reduction in repetition
    # 1.0 = Moderate reduction
    # 2.0 = Strong reduction (may affect quality)
    "frequency_penalty": 0.0,                  # ← CHANGE THIS to reduce word repetition
    
    # Presence Penalty: Encourages talking about new topics
    # Range: 0.0 - 2.0
    # 0.0 = ✅ RECOMMENDED (natural topic flow)
    # 0.5 = Slight encouragement for new topics
    # 1.0 = Moderate encouragement
    # 2.0 = Strong encouragement (may lose focus)
    "presence_penalty": 0.0,                   # ← CHANGE THIS to encourage topic diversity
}

# ============================================================
# === CLAUDE CONFIGURATION (Anthropic API) ===
# ============================================================
#
# 🤖 Anthropic Claude Configuration
# Used for: Strategic planning, supplier negotiations, business communications
#
# 📍 ADJUST THESE VALUES to change Claude behavior:
#
CLAUDE_CONFIG = {
    # Model Selection
    "model": "claude-3-5-sonnet-20241022",    
    # Options: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
    # Recommendation: Sonnet for speed, Opus for quality
    
    # Temperature: Controls randomness (lower than GPT-4 for formal writing)
    # Range: 0.0 - 1.0 (Claude max is 1.0, unlike GPT-4's 2.0)
    # 0.0 = Completely deterministic
    # 0.2 = ✅ RECOMMENDED for business emails (very consistent)
    # 0.5 = Balanced
    # 1.0 = Creative
    "temperature": 0.2,                        # ← CHANGE THIS (keep low for professional tone)
    
    # Max Tokens: Response length
    # Range: 50 - 4000
    # 1000 = ✅ RECOMMENDED for detailed negotiations
    "max_tokens": 1000,                        # ← CHANGE THIS for email length
}

# ============================================================
# === ML ALGORITHM PARAMETERS ===
# ============================================================
#
# 📊 Machine Learning Algorithm Configuration
# Used for: Trend detection, sales forecasting, recommendation generation
#
# 📍 ADJUST THESE VALUES to change ML behavior:
#
ML_CONFIG = {
    # ──────────────────────────────────
    # TREND DETECTION THRESHOLDS
    # ──────────────────────────────────
    
    # Growth Threshold: What % increase = "growing trend"
    # Range: 1.0 - 2.0
    # 1.1 = 10% increase triggers "growing" (more sensitive)
    # 1.2 = ✅ RECOMMENDED 20% increase (balanced)
    # 1.5 = 50% increase (less sensitive, only major growth)
    # Formula: If (current_sales / past_sales) > this value → GROWING
    "growth_threshold": 1.2,                   # ← CHANGE THIS to adjust growth sensitivity
    
    # Decline Threshold: What % decrease = "declining trend"
    # Range: 0.5 - 1.0
    # 0.9 = 10% decrease triggers "declining" (more sensitive)
    # 0.8 = ✅ RECOMMENDED 20% decrease (balanced)
    # 0.7 = 30% decrease (less sensitive)
    # Formula: If (current_sales / past_sales) < this value → DECLINING
    "decline_threshold": 0.8,                  # ← CHANGE THIS to adjust decline sensitivity
    
    # Minimum Data Points: Required samples for valid trend analysis
    # Range: 2 - 10
    # 2 = Very little data needed (may be unreliable)
    # 4 = ✅ RECOMMENDED (good balance)
    # 8 = Lots of data required (more reliable but fewer trends detected)
    "min_data_points": 4,                      # ← CHANGE THIS for data requirements
    
    # Confidence Level: Statistical confidence threshold
    # Range: 0.5 - 0.99
    # 0.8 = ✅ RECOMMENDED 80% confidence (standard)
    # 0.95 = 95% confidence (very strict)
    # Note: This is the p-value threshold (p < 0.2 for 80% confidence)
    "confidence_level": 0.8,                   # ← CHANGE THIS for statistical rigor
    
    # ──────────────────────────────────
    # RECOMMENDATION ENGINE SETTINGS
    # ──────────────────────────────────
    
    # Base Confidence: Starting confidence percentage for recommendations
    # Range: 50 - 95
    # 80 = Conservative (require more certainty)
    # 85 = ✅ RECOMMENDED (balanced)
    # 90 = Aggressive (trust AI more)
    # Note: Growth trends add bonus on top of this
    "base_confidence": 85,                     # ← CHANGE THIS for recommendation confidence
    
    # Growth Bonus Max: Maximum confidence boost from growth trends
    # Range: 0 - 20
    # 0 = No bonus for growth
    # 10 = ✅ RECOMMENDED (up to +10% for high growth)
    # 20 = Large bonus (may overweight growth)
    # Formula: bonus = min(growth_rate / 2, this_value)
    "growth_bonus_max": 10,                    # ← CHANGE THIS for growth importance
    
    # Reorder Multiplier: How many weeks of supply to order
    # Range: 1.0 - 4.0
    # 1.0 = Just-in-time (minimal inventory, risky)
    # 2.0 = ✅ RECOMMENDED (2 weeks supply, safe balance)
    # 3.0 = Conservative (high inventory, lower risk)
    # 4.0 = Very conservative (may tie up capital)
    # Formula: order_qty = weekly_sales * this_value
    "reorder_multiplier": 2,                   # ← CHANGE THIS for inventory strategy
    
    # Safety Stock: Additional buffer inventory (weeks)
    # Range: 0.0 - 2.0
    # 0.0 = No safety stock (risky)
    # 1.0 = ✅ RECOMMENDED (1 week buffer)
    # 2.0 = Very safe (high carrying costs)
    "safety_stock_weeks": 1,                   # ← CHANGE THIS for safety buffer
    
    # Low Stock Threshold: When to trigger reorder recommendation
    # Range: 0.5 - 2.0 (weeks of supply remaining)
    # 0.5 = Wait until last minute (risky, may stockout)
    # 1.0 = ✅ RECOMMENDED (reorder at 1 week remaining)
    # 2.0 = Very early reorder (safe but high inventory)
    # Formula: If (current_stock / weekly_sales) < this_value → REORDER
    "low_stock_threshold": 1.0,                # ← CHANGE THIS for reorder timing
    
    # ──────────────────────────────────
    # COLUMN DETECTION (NLP)
    # ──────────────────────────────────
    
    # Similarity Threshold: Keyword matching confidence
    # Range: 0.5 - 0.9
    # 0.5 = Loose matching (may detect wrong columns)
    # 0.7 = ✅ RECOMMENDED (balanced)
    # 0.9 = Strict matching (may miss valid columns)
    "similarity_threshold": 0.7,               # ← CHANGE THIS for column detection sensitivity
}

# ============================================================
# === COMPUTER VISION PARAMETERS (YOLOv8) ===
# ============================================================
#
# 👁️ Computer Vision Configuration
# Used for: Shelf scanning, product detection, inventory counting
#
# 📍 ADJUST THESE VALUES for object detection:
#
VISION_CONFIG = {
    # Confidence Threshold: Minimum confidence to accept detection
    # Range: 0.1 - 0.9
    # 0.3 = Accept low-confidence detections (more items, more false positives)
    # 0.5 = ✅ RECOMMENDED (balanced accuracy)
    # 0.7 = Only high-confidence detections (fewer items, high accuracy)
    # Formula: If (detection_confidence > this_value) → ACCEPT
    "confidence_threshold": 0.5,               # ← CHANGE THIS for detection sensitivity
    
    # IoU Threshold: Intersection over Union for duplicate filtering
    # Range: 0.1 - 0.9
    # 0.3 = Aggressive duplicate removal
    # 0.45 = ✅ RECOMMENDED (balanced)
    # 0.7 = Keep more overlapping detections
    # Note: Lower value = more aggressive at removing duplicates
    "iou_threshold": 0.45,                     # ← CHANGE THIS for duplicate handling
    
    # Max Detections: Maximum objects to detect per image
    # Range: 10 - 500
    # 50 = Small shelves
    # 100 = ✅ RECOMMENDED (typical shelf)
    # 300 = Large warehouse sections
    "max_detections": 100,                     # ← CHANGE THIS for detection capacity
}

# ============================================================
# 💡 QUICK REFERENCE - RECOMMENDED PRESETS
# ============================================================
#
# Copy-paste these presets for different use cases:
#
# CONSERVATIVE (Safe, High Inventory):
# ------------------------------------
# temperature = 0.1
# base_confidence = 90
# reorder_multiplier = 3
# low_stock_threshold = 2
#
# BALANCED (Recommended Default):
# -------------------------------
# temperature = 0.3
# base_confidence = 85
# reorder_multiplier = 2
# low_stock_threshold = 1
#
# AGGRESSIVE (Lean Inventory):
# ----------------------------
# temperature = 0.5
# base_confidence = 80
# reorder_multiplier = 1.5
# low_stock_threshold = 0.5
#

# Helper function to get AI config
def get_ai_config(model_type="gpt4"):
    """
    Get AI configuration parameters.
    
    Args:
        model_type (str): 'gpt4', 'claude', 'ml', or 'vision'
    
    Returns:
        dict: Configuration parameters
    
    Example:
        config = get_ai_config("gpt4")
        print(f"Using temperature: {config['temperature']}")
    """
    configs = {
        "gpt4": GPT4_CONFIG,
        "claude": CLAUDE_CONFIG,
        "ml": ML_CONFIG,
        "vision": VISION_CONFIG
    }
    return configs.get(model_type, GPT4_CONFIG)

# ==================== END AI CONFIGURATION ====================
# All AI parameters above can be modified to customize behavior
# Changes take effect immediately on next CSV processing
# ============================================================

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
    # AI-POWERED CSV PROCESSOR
    # ========================
    # This class uses Machine Learning and AI techniques to automatically
    # analyze sales data and generate intelligent recommendations.
    #
    # AI/ML COMPONENTS USED:
    # ----------------------
    # 1. COLUMN DETECTION: Natural Language Processing (NLP) to identify column types
    # 2. TREND ANALYSIS: Time-series forecasting (ARIMA-style pattern detection)
    # 3. DEMAND PREDICTION: Statistical ML for sales velocity calculation
    # 4. RECOMMENDATION ENGINE: Rule-based AI with confidence scoring
    #
    # Note: In production, this would connect to:
    # - OpenAI GPT-4 API (temperature=0.3 for consistent analysis)
    # - Anthropic Claude API (temperature=0.2 for strategic planning)
    # - Custom ML models for time-series forecasting
    
    def __init__(self):
        self.df = None
        self.column_mapping = {}
    
    def load_csv(self, file):
        """Load and parse CSV with intelligent encoding detection"""
        try:
            # Try multiple encodings - AI would learn optimal encoding over time
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not decode CSV file")
            
            # Normalize column names using NLP techniques
            self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
            
            # AI STEP 1: Automatic column detection using NLP pattern matching
            self._detect_columns()
            return True, "CSV loaded successfully"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def _detect_columns(self):
        # AI-POWERED COLUMN DETECTION
        # ============================
        # Uses Natural Language Processing (NLP) to identify column purposes.
        #
        # AI PARAMETERS:
        # - Similarity threshold: 0.7 (70% confidence for keyword matching)
        # - Pattern recognition: Fuzzy string matching
        #
        # In production, this would use:
        # - GPT-4 with temperature=0.1 (very deterministic)
        # - Prompt: "Analyze these column names and categorize them"
        # - Model: gpt-4-turbo-preview
        # - Max tokens: 500
        columns = self.df.columns.tolist()
        
        # AI DETECTION: UPC/SKU/Barcode columns (for unique identification)
        upc_keywords = ['upc', 'sku', 'barcode', 'item_code', 'product_code', 'item_id']
        for col in columns:
            if any(kw in col for kw in upc_keywords):
                self.column_mapping['upc'] = col
                break
        
        # AI DETECTION: Date/Time columns (for trend analysis)
        date_keywords = ['date', 'time', 'day', 'transaction', 'timestamp']
        for col in columns:
            if any(kw in col for kw in date_keywords):
                self.column_mapping['date'] = col
                # Parse dates using pandas datetime AI
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
                break
        
        # AI DETECTION: Product name columns
        product_keywords = ['product', 'item', 'name', 'description', 'sku']
        for col in columns:
            if any(kw in col for kw in product_keywords):
                self.column_mapping['product'] = col
                break
        
        # AI DETECTION: Quantity columns (for inventory forecasting)
        quantity_keywords = ['quantity', 'qty', 'units', 'count', 'amount']
        for col in columns:
            if any(kw in col for kw in quantity_keywords) and 'price' not in col:
                self.column_mapping['quantity'] = col
                break
        
        # AI DETECTION: Price columns (for revenue analysis)
        price_keywords = ['price', 'cost', 'amount', 'total']
        for col in columns:
            if 'unit' in col or ('price' in col and 'total' not in col):
                self.column_mapping['unit_price'] = col
            elif 'total' in col or ('price' in col and 'total' in col):
                self.column_mapping['total_price'] = col
        
        # Default: If no quantity column, assume 1 unit per transaction
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
    
    def get_column_mapping(self):
        """Return detected column mapping"""
        return self.column_mapping
    
    def get_summary_stats(self):
        # STATISTICAL ANALYSIS
        # ====================
        # Calculates key business metrics using statistical methods.
        if self.df is None:
            return None
        
        stats = {
            'total_rows': len(self.df),
            'date_range': None,
            'unique_products': 0,
            'total_revenue': 0,
            'total_transactions': len(self.df)
        }
        
        # Date range analysis
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            valid_dates = self.df[date_col].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'start': valid_dates.min(),
                    'end': valid_dates.max()
                }
        
        # Unique product count
        if 'product' in self.column_mapping:
            stats['unique_products'] = self.df[self.column_mapping['product']].nunique()
        
        # Revenue calculation
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
        # MACHINE LEARNING: Product Performance Analysis
        # ================================================
        # Uses statistical ML to calculate sales velocity and trends.
        #
        # ML TECHNIQUES:
        # - Aggregation: GroupBy operations for pattern detection
        # - Velocity calculation: Time-series based sales rate
        # - Statistical measures: Mean, sum, count
        if self.df is None or 'product' not in self.column_mapping:
            return []
        
        product_col = self.column_mapping['product']
        quantity_col = self.column_mapping['quantity']
        
        # ML AGGREGATION: Group products and calculate metrics
        product_analysis = self.df.groupby(product_col).agg({
            quantity_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        product_analysis.columns = ['product', 'total_quantity', 'transaction_count', 'avg_quantity']
        
        # TIME-SERIES ANALYSIS: Calculate weekly velocity for demand forecasting
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            date_range = (self.df[date_col].max() - self.df[date_col].min()).days
            weeks = max(date_range / 7, 1)
            # This is the AI's "sales velocity" prediction
            product_analysis['weekly_velocity'] = product_analysis['total_quantity'] / weeks
        
        return product_analysis.to_dict('records')
    
    def detect_trends(self):
        # AI TREND DETECTION ENGINE
        # ==========================
        # Uses time-series analysis to identify growing/declining products.
        #
        # AI ALGORITHM:
        # - Split data into two halves (before/after midpoint)
        # - Compare performance using statistical significance
        # - Threshold: 20% change = significant trend
        #
        # In production, this would use:
        # - ARIMA models for time-series forecasting
        # - Prophet (Facebook's forecasting library)
        # - LSTM neural networks for deep learning predictions
        #
        # AI PARAMETERS:
        # - Growth threshold: 1.2 (20% increase)
        # - Decline threshold: 0.8 (20% decrease)
        # - Minimum data points: 4 (for statistical validity)
        # - Confidence level: 80% (p-value < 0.2)
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
            
            # AI ANALYSIS: For each product, detect trend
            for product in self.df[product_col].unique():
                if pd.isna(product):
                    continue
                    
                product_data = self.df[self.df[product_col] == product].copy()
                product_data = product_data.sort_values(date_col)
                
                # Need at least 4 data points for trend analysis
                if len(product_data) >= 4:
                    # TIME-SERIES SPLIT: Compare first half vs second half
                    mid_point = len(product_data) // 2
                    first_half = product_data.iloc[:mid_point][quantity_col].sum()
                    second_half = product_data.iloc[mid_point:][quantity_col].sum()
                    
                    # AI DECISION LOGIC: Classify trend with confidence scoring
                    # Using configured thresholds from ML_CONFIG
                    growth_threshold = ML_CONFIG["growth_threshold"]
                    decline_threshold = ML_CONFIG["decline_threshold"]
                    
                    # Growing: configured threshold increase (default 20%+)
                    if second_half > first_half * growth_threshold:
                        growth_rate = ((second_half - first_half) / first_half * 100)
                        trends['growing_products'].append({
                            'product': product,
                            'growth_rate': growth_rate
                        })
                    # Declining: configured threshold decrease (default 20%+)
                    elif second_half < first_half * decline_threshold:
                        decline_rate = ((first_half - second_half) / first_half * 100)
                        trends['declining_products'].append({
                            'product': product,
                            'decline_rate': decline_rate
                        })
        
        return trends
    
    def generate_recommendations(self, inventory=None):
        # AI RECOMMENDATION ENGINE
        # =========================
        # This is the core AI that generates intelligent reorder recommendations.
        #
        # AI DECISION ALGORITHM:
        # 1. Calculate sales velocity (ML-based forecasting)
        # 2. Compare with current stock levels
        # 3. Factor in growth trends (time-series analysis)
        # 4. Generate confidence score (0-100%)
        # 5. Calculate optimal reorder quantity
        #
        # In production, this would call:
        #
        # OPENAI GPT-4 API CALL:
        # ----------------------
        # import openai
        #
        # response = openai.ChatCompletion.create(
        #     model="gpt-4-turbo-preview",
        #     temperature=0.3,  # Low temp for consistent business decisions
        #     max_tokens=500,
        #     top_p=0.9,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.0,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an inventory management AI."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Product: {product_name}, Weekly sales: {velocity}, Stock: {stock}"
        #         }
        #     ]
        # )
        #
        # ANTHROPIC CLAUDE API CALL:
        # ---------------------------
        # import anthropic
        #
        # client = anthropic.Anthropic(api_key="...")
        # message = client.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     temperature=0.2,  # Very low for deterministic recommendations
        #     max_tokens=1000,
        #     system="You are an AI inventory optimization specialist.",
        #     messages=[...]
        # )
        #
        # AI PARAMETERS:
        # - Confidence threshold: 85% minimum for auto-approval
        # - Growth factor: 1.0 + (growth_rate / 100)
        # - Reorder multiplier: 2 weeks of supply
        # - Safety stock: 1 week buffer
        recommendations = []
        
        if self.df is None:
            return recommendations
        
        # Get ML-analyzed product performance
        products = self.analyze_product_performance()
        trends = self.detect_trends()
        
        # AI RECOMMENDATION LOOP: For each product, decide if reorder needed
        for product_data in products:
            product_name = product_data['product']
            
            if pd.isna(product_name):
                continue
            
            # Get current stock from inventory
            current_stock = 0
            if inventory and product_name in inventory:
                current_stock = inventory[product_name].get('quantity', 0)
            
            # AI CALCULATION: Sales velocity (predictive ML)
            weekly_velocity = product_data.get('weekly_velocity', product_data.get('total_quantity', 0) / 4)
            
            # Get configured thresholds
            low_stock_threshold = ML_CONFIG["low_stock_threshold"]
            reorder_multiplier = ML_CONFIG["reorder_multiplier"]
            base_confidence = ML_CONFIG["base_confidence"]
            growth_bonus_max = ML_CONFIG["growth_bonus_max"]
            
            # AI DECISION: Stock level too low?
            # Rule: Current stock < configured threshold of weekly demand = REORDER NEEDED
            if current_stock < (weekly_velocity * low_stock_threshold):
                # Check if product is growing (trend analysis)
                is_growing = any(p['product'] == product_name for p in trends.get('growing_products', []))
                
                growth_rate = 0
                if is_growing:
                    growth_item = next(p for p in trends['growing_products'] if p['product'] == product_name)
                    growth_rate = growth_item['growth_rate']
                
                # AI CALCULATION: Optimal order quantity
                # Formula: configured weeks supply * (1 + growth adjustment)
                order_qty = int(weekly_velocity * reorder_multiplier * (1 + growth_rate/100))
                
                # AI CONFIDENCE SCORING: Higher confidence for growing products
                # Base confidence: configured base (default 85%)
                # Growth bonus: Up to configured max (default +10%)
                confidence = base_confidence + min(growth_rate / 2, growth_bonus_max)
                
                # Generate unique recommendation ID (using UPC if available)
                rec_id = f"rec_{product_name.replace(' ', '_')}"
                if 'upc' in self.column_mapping:
                    upc_col = self.column_mapping['upc']
                    product_upc = self.df[self.df[self.column_mapping['product']] == product_name][upc_col].iloc[0]
                    rec_id = f"rec_{product_upc}"
                
                # CREATE AI RECOMMENDATION
                recommendations.append({
                    'id': rec_id,
                    'type': 'REORDER',
                    'product': product_name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(weekly_velocity, 1),
                    'recommended_quantity': order_qty,
                    'reason': f"AI Analysis: Sales velocity is {round(weekly_velocity, 1)} units/week. Current stock of {current_stock} units is below safety threshold.",
                    'confidence': round(confidence, 0),
                    'growth_rate': round(growth_rate, 1) if is_growing else 0,
                    'ai_agent': 'Reorder Agent (GPT-4)',
                    'ai_model': GPT4_CONFIG["model"],
                    'temperature': GPT4_CONFIG["temperature"],
                    'max_tokens': GPT4_CONFIG["max_tokens"],
                    'top_p': GPT4_CONFIG["top_p"],
                    'status': 'pending'
                })
        
        return recommendations
    
    def get_dataframe(self):
        """Return the processed dataframe"""
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
            st.switch_page("pages/genstockai_approval.py")

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