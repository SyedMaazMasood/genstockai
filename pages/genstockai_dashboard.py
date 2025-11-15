import streamlit as st

st.title("📈 Dashboard")

# Key Metrics
st.markdown("### Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Items to Review",
        value="3"
    )

with col2:
    st.metric(
        label="Est. Weekly Time Saved",
        value="7 Hours"
    )

with col3:
    st.metric(
        label="Est. Monthly Waste Reduced",
        value="$500+"
    )

st.markdown("---")

# Success message
st.success("Welcome! You have 3 new recommendations in your Approval Queue.")

st.markdown("---")

# Additional dashboard content
st.markdown("### Recent Activity")

with st.container(border=True):
    st.markdown("**Today**")
    st.markdown("- 🔄 3 new recommendations generated")
    st.markdown("- 📊 Sales data analyzed (48 hours)")
    st.markdown("- ✅ 2 orders automatically approved")

with st.container(border=True):
    st.markdown("**This Week**")
    st.markdown("- 📈 15 recommendations processed")
    st.markdown("- 💰 $1,200 saved through waste reduction")
    st.markdown("- ⏱️ 7 hours of manual work automated")

st.markdown("---")

# Quick navigation
st.markdown("### Quick Navigation")
col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Approval Queue →", use_container_width=True, key="dash_queue"):
        st.switch_page("pages/3_✅_Approval_Queue.py")

with col2:
    if st.button("Upload New Data →", use_container_width=True, key="dash_upload"):
        st.switch_page("pages/2_📤_Data_Sources.py")

st.markdown("---")

# Add AI Processing Details
st.markdown("### 🤖 AI Processing Pipeline")
with st.container(border=True):
    st.markdown("**Current AI Models Active:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**NLP & Analysis:**")
        st.markdown("- GPT-4 for demand forecasting")
        st.markdown("- Claude for negotiation drafting")
        st.markdown("- Custom ML for pattern detection")
    
    with col2:
        st.markdown("**Computer Vision:**")
        st.markdown("- YOLOv8 for shelf scanning")
        st.markdown("- OCR for receipt processing")
        st.markdown("- Image classification for products")