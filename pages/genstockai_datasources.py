import streamlit as st

st.title("📤 Data Sources")

st.markdown("Connect your data sources to enable AI-powered recommendations.")

st.markdown("---")

# POS API Connection
st.subheader("POS Connection (Real-time)")
st.markdown("Connect your Point of Sale system for automatic data sync.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Connect to Clover POS", use_container_width=True, type="primary"):
        st.info("🔄 Clover POS integration coming soon!")

with col2:
    if st.button("Connect to Square POS", use_container_width=True, type="primary"):
        st.info("🔄 Square POS integration coming soon!")

st.markdown("---")

# CSV Upload
st.subheader("Manual Upload")
st.markdown("Upload your sales report in CSV format for analysis.")

uploaded_file = st.file_uploader("Upload your CSV Sales Report", type=["csv"])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.info("💡 AI agents will process this data and generate recommendations within minutes.")
    
    if st.button("Process Data Now"):
        with st.spinner("Processing your sales data..."):
            import time
            time.sleep(2)
        st.success("✅ Data processed successfully! Check your Approval Queue for new recommendations.")
        if st.button("Go to Approval Queue →"):
            st.switch_page("pages/3_✅_Approval_Queue.py")

st.markdown("---")

# Photo Scan
st.subheader("Shelf Scan")
st.markdown("Take a photo of your shelves to track physical inventory levels.")

camera_photo = st.camera_input("Scan your shelf stock")

if camera_photo is not None:
    st.success("✅ Photo captured successfully!")
    st.info("💡 AI vision will analyze stock levels and update inventory automatically.")
    
    if st.button("Analyze Photo"):
        with st.spinner("Analyzing shelf stock..."):
            import time
            time.sleep(2)
        st.success("✅ Analysis complete! Detected 12 items with low stock.")

st.markdown("---")

# Data source status
st.subheader("Connected Sources")

status_data = [
    {"Source": "CSV Upload", "Status": "✅ Active", "Last Sync": "2 days ago"},
    {"Source": "Clover POS", "Status": "⚪ Not Connected", "Last Sync": "N/A"},
    {"Source": "Square POS", "Status": "⚪ Not Connected", "Last Sync": "N/A"},
    {"Source": "Shelf Scanner", "Status": "⚪ Not Configured", "Last Sync": "N/A"},
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