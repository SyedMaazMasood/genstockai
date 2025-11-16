import json
import os
from datetime import datetime

# File paths for persistent storage
DATA_DIR = "data"
SALES_DATA_FILE = os.path.join(DATA_DIR, "sales_data.json")
VENDORS_FILE = os.path.join(DATA_DIR, "vendors.json")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Default vendor data structure
DEFAULT_VENDORS = [
    {
        "id": "vendor_1",
        "name": "Beverage Distributors Inc.",
        "contact": "+1-555-0123",
        "email": "orders@bevdist.com",
        "order_method": "email",  # email, phone, whatsapp, website, manual
        "order_frequency": "weekly",  # weekly, biweekly, monthly, as_needed
        "order_day": "Monday",
        "products": ["Red Bull", "Coca-Cola", "Pepsi", "Energy Drinks"],
        "lead_time_days": 2,
        "minimum_order": 50.00
    },
    {
        "id": "vendor_2",
        "name": "Fresh Bakery Supply",
        "contact": "+1-555-0456",
        "email": "orders@freshbakery.com",
        "order_method": "phone",
        "order_frequency": "daily",
        "order_day": "Daily",
        "products": ["Croissants", "Muffins", "Bagels", "Bread"],
        "lead_time_days": 1,
        "minimum_order": 30.00
    },
    {
        "id": "vendor_3",
        "name": "Peak Coffee",
        "contact": "+1-555-0789",
        "email": "sales@peakcoffee.com",
        "order_method": "website",
        "order_frequency": "biweekly",
        "order_day": "Wednesday",
        "products": ["Coffee Beans", "Tea", "Syrups"],
        "lead_time_days": 3,
        "minimum_order": 100.00
    }
]

# Load or initialize vendors
def load_vendors():
    if os.path.exists(VENDORS_FILE):
        with open(VENDORS_FILE, 'r') as f:
            return json.load(f)
    else:
        save_vendors(DEFAULT_VENDORS)
        return DEFAULT_VENDORS

def save_vendors(vendors):
    with open(VENDORS_FILE, 'w') as f:
        json.dump(vendors, f, indent=2)

# Load or initialize inventory
def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_inventory(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)

# Load or initialize sales data
def load_sales_data():
    if os.path.exists(SALES_DATA_FILE):
        with open(SALES_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_sales_data(sales_data):
    with open(SALES_DATA_FILE, 'w') as f:
        json.dump(sales_data, f, indent=2)

# Load or initialize recommendations
def load_recommendations():
    if os.path.exists(RECOMMENDATIONS_FILE):
        with open(RECOMMENDATIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_recommendations(recommendations):
    with open(RECOMMENDATIONS_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2)

# CSV column mapping suggestions (AI will detect these)
COMMON_COLUMN_NAMES = {
    'date': ['date', 'transaction_date', 'sale_date', 'datetime', 'timestamp'],
    'product': ['product', 'item', 'product_name', 'item_name', 'description'],
    'quantity': ['quantity', 'qty', 'units', 'amount', 'count'],
    'price': ['price', 'unit_price', 'cost', 'amount', 'total'],
    'total': ['total', 'total_price', 'total_amount', 'sale_amount'],
    'category': ['category', 'type', 'product_type', 'item_category']
}