import streamlit as st
import json
import os
import uuid

# ==================== EMBEDDED CONFIG ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

VENDORS_FILE = os.path.join(DATA_DIR, "vendors.json")

DEFAULT_VENDORS = [
    {
        "id": "vendor_1",
        "name": "Beverage Distributors Inc.",
        "contact": "+1-555-0123",
        "email": "orders@bevdist.com",
        "order_method": "email",
        "order_frequency": "weekly",
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

def load_vendors():
    if os.path.exists(VENDORS_FILE):
        with open(VENDORS_FILE, 'r') as f:
            return json.load(f)
    else:
        with open(VENDORS_FILE, 'w') as f:
            json.dump(DEFAULT_VENDORS, f, indent=2)
        return DEFAULT_VENDORS

def save_vendors(vendors):
    with open(VENDORS_FILE, 'w') as f:
        json.dump(vendors, f, indent=2)

# ==================== MAIN PAGE CODE ====================
st.title("🏪 Vendor Management")
st.markdown("Manage your suppliers, delivery schedules, and ordering preferences.")

st.markdown("---")

vendors = load_vendors()

tab1, tab2 = st.tabs(["📋 All Vendors", "➕ Add New Vendor"])

with tab1:
    st.markdown("### Your Vendors")
    
    if not vendors:
        st.info("No vendors configured yet. Add your first vendor in the 'Add New Vendor' tab!")
    
    for i, vendor in enumerate(vendors):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {vendor['name']}")
                st.markdown(f"**Contact:** {vendor.get('email', '')} | {vendor.get('contact', '')}")
            
            with col2:
                if st.button("✏️ Edit", key=f"edit_{vendor['id']}"):
                    st.session_state[f'editing_{vendor["id"]}'] = True
                if st.button("🗑️ Delete", key=f"delete_{vendor['id']}"):
                    vendors = [v for v in vendors if v['id'] != vendor['id']]
                    save_vendors(vendors)
                    st.success("Vendor deleted!")
                    st.rerun()
            
            if st.session_state.get(f'editing_{vendor["id"]}', False):
                st.markdown("---")
                st.markdown("#### Edit Vendor Details")
                
                with st.form(key=f"edit_form_{vendor['id']}"):
                    vendor['name'] = st.text_input("Vendor Name", value=vendor['name'])
                    vendor['contact'] = st.text_input("Phone Number", value=vendor.get('contact', ''))
                    vendor['email'] = st.text_input("Email", value=vendor.get('email', ''))
                    
                    vendor['order_method'] = st.selectbox(
                        "Ordering Method",
                        options=['email', 'phone', 'whatsapp', 'website', 'manual'],
                        index=['email', 'phone', 'whatsapp', 'website', 'manual'].index(vendor.get('order_method', 'email'))
                    )
                    
                    vendor['order_frequency'] = st.selectbox(
                        "Order Frequency",
                        options=['daily', 'weekly', 'biweekly', 'monthly', 'as_needed'],
                        index=['daily', 'weekly', 'biweekly', 'monthly', 'as_needed'].index(vendor.get('order_frequency', 'weekly'))
                    )
                    
                    if vendor['order_frequency'] in ['weekly', 'biweekly']:
                        vendor['order_day'] = st.selectbox(
                            "Order Day",
                            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(vendor.get('order_day', 'Monday'))
                        )
                    
                    vendor['lead_time_days'] = st.number_input("Lead Time (days)", min_value=0, value=vendor.get('lead_time_days', 2))
                    vendor['minimum_order'] = st.number_input("Minimum Order ($)", min_value=0.0, value=float(vendor.get('minimum_order', 0)))
                    
                    products_text = st.text_area(
                        "Products (one per line)",
                        value='\n'.join(vendor.get('products', []))
                    )
                    vendor['products'] = [p.strip() for p in products_text.split('\n') if p.strip()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("💾 Save Changes", type="primary"):
                            for j, v in enumerate(vendors):
                                if v['id'] == vendor['id']:
                                    vendors[j] = vendor
                            save_vendors(vendors)
                            st.session_state[f'editing_{vendor["id"]}'] = False
                            st.success("Vendor updated!")
                            st.rerun()
                    
                    with col2:
                        if st.form_submit_button("❌ Cancel"):
                            st.session_state[f'editing_{vendor["id"]}'] = False
                            st.rerun()
            
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Order Method:**")
                    method_icons = {
                        'email': '📧',
                        'phone': '📞',
                        'whatsapp': '💬',
                        'website': '🌐',
                        'manual': '📝'
                    }
                    method = vendor.get('order_method', 'email')
                    st.markdown(f"{method_icons.get(method, '📋')} {method.title()}")
                
                with col2:
                    st.markdown("**Frequency:**")
                    st.markdown(f"🔄 {vendor.get('order_frequency', 'weekly').title()}")
                    if vendor.get('order_day'):
                        st.markdown(f"📅 {vendor['order_day']}")
                
                with col3:
                    st.markdown("**Lead Time:**")
                    st.markdown(f"⏱️ {vendor.get('lead_time_days', 2)} days")
                
                with st.expander(f"📦 Products ({len(vendor.get('products', []))})"):
                    for product in vendor.get('products', []):
                        st.markdown(f"- {product}")

with tab2:
    st.markdown("### Add New Vendor")
    
    with st.form("add_vendor_form"):
        new_vendor = {}
        
        new_vendor['name'] = st.text_input("Vendor Name *", placeholder="e.g., Fresh Produce Co.")
        
        col1, col2 = st.columns(2)
        with col1:
            new_vendor['contact'] = st.text_input("Phone Number", placeholder="+1-555-0123")
        with col2:
            new_vendor['email'] = st.text_input("Email *", placeholder="orders@vendor.com")
        
        st.markdown("---")
        st.markdown("#### Ordering Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_vendor['order_method'] = st.selectbox(
                "How do you order? *",
                options=['email', 'phone', 'whatsapp', 'website', 'manual'],
                format_func=lambda x: {
                    'email': '📧 Email',
                    'phone': '📞 Phone Call',
                    'whatsapp': '💬 WhatsApp',
                    'website': '🌐 Website/Portal',
                    'manual': '📝 Manual (print & hand deliver)'
                }[x]
            )
        
        with col2:
            new_vendor['order_frequency'] = st.selectbox(
                "How often do you order? *",
                options=['daily', 'weekly', 'biweekly', 'monthly', 'as_needed'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        if new_vendor['order_frequency'] in ['weekly', 'biweekly']:
            new_vendor['order_day'] = st.selectbox(
                "Which day?",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            )
        else:
            new_vendor['order_day'] = "N/A"
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_vendor['lead_time_days'] = st.number_input(
                "Lead Time (days)",
                min_value=0,
                value=2,
                help="How many days between order and delivery?"
            )
        
        with col2:
            new_vendor['minimum_order'] = st.number_input(
                "Minimum Order ($)",
                min_value=0.0,
                value=50.0,
                step=10.0
            )
        
        st.markdown("---")
        st.markdown("#### Products")
        
        products_text = st.text_area(
            "What products do they supply? (one per line) *",
            placeholder="Red Bull\nCoca-Cola\nPepsi\nEnergy Drinks",
            height=150
        )
        
        if st.form_submit_button("➕ Add Vendor", type="primary"):
            if not new_vendor['name'] or not new_vendor['email']:
                st.error("Please fill in all required fields (*)")
            elif not products_text.strip():
                st.error("Please add at least one product")
            else:
                new_vendor['id'] = f"vendor_{uuid.uuid4().hex[:8]}"
                new_vendor['products'] = [p.strip() for p in products_text.split('\n') if p.strip()]
                
                vendors.append(new_vendor)
                save_vendors(vendors)
                
                st.success(f"✅ Vendor '{new_vendor['name']}' added successfully!")
                st.balloons()
                
                import time
                time.sleep(1)
                st.rerun()

st.markdown("---")

st.markdown("### 📊 Vendor Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Vendors", len(vendors))

with col2:
    methods = {}
    for v in vendors:
        method = v.get('order_method', 'email')
        methods[method] = methods.get(method, 0) + 1
    most_common = max(methods.items(), key=lambda x: x[1])[0] if methods else 'N/A'
    st.metric("Most Common Method", most_common.title())

with col3:
    all_products = set()
    for v in vendors:
        all_products.update(v.get('products', []))
    st.metric("Total Products", len(all_products))