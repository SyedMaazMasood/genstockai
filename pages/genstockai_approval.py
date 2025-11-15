import streamlit as st

st.title("✅ Approval Queue (3)")

st.markdown("Review AI-generated recommendations and approve or reject with one tap.")

st.markdown("---")

# Mock recommendations data
recommendations = [
    {
        "id": 1,
        "type": "REORDER",
        "title": "REORDER: Red Bull",
        "recommendation": "Order 2 cases (48 units)",
        "reason": "Sales are up 20% on Fridays, and current stock is low.",
        "icon": "🔄",
        "details": {
            "Current Stock": "8 units",
            "Avg Weekly Sales": "36 units",
            "Estimated Cost": "$96.00",
            "Supplier": "Beverage Distributors Inc."
        }
    },
    {
        "id": 2,
        "type": "PROMOTION",
        "title": "PROMOTION: Croissants",
        "recommendation": "Create 50% off 'end-of-day' deal",
        "reason": "Stock is high (20 units) and they expire tomorrow. This avoids waste.",
        "icon": "🏷️",
        "details": {
            "Current Stock": "20 units",
            "Expiration": "Tomorrow (11/16)",
            "Normal Price": "$3.50 each",
            "Promo Price": "$1.75 each",
            "Est. Revenue Recovery": "$35.00"
        }
    },
    {
        "id": 3,
        "type": "NEGOTIATE",
        "title": "NEGOTIATE: Coffee Beans",
        "recommendation": "Draft message to 'Peak Coffee' asking for 5% discount",
        "reason": "Competitor 'Roast Co.' is offering a 5% lower price this week.",
        "icon": "💬",
        "details": {
            "Current Supplier": "Peak Coffee",
            "Current Price": "$12.50/lb",
            "Competitor Price": "$11.88/lb (Roast Co.)",
            "Monthly Volume": "40 lbs",
            "Potential Savings": "$24.80/month"
        }
    }
]

# Initialize session state for tracking approvals/rejections
if 'approved' not in st.session_state:
    st.session_state.approved = []
if 'rejected' not in st.session_state:
    st.session_state.rejected = []

# Display each recommendation
for rec in recommendations:
    # Skip if already processed
    if rec['id'] in st.session_state.approved or rec['id'] in st.session_state.rejected:
        continue
    
    with st.container(border=True):
        # Header with icon and title
        st.markdown(f"## {rec['icon']} {rec['title']}")
        
        # Recommendation
        st.markdown(f"### 📋 Recommendation")
        st.markdown(f"**{rec['recommendation']}**")
        
        # Explain Why section
        with st.expander("🤔 Explain Why?", expanded=False):
            st.markdown(f"**Reason:** {rec['reason']}")
            st.markdown("---")
            st.markdown("**Additional Details:**")
            for key, value in rec['details'].items():
                st.markdown(f"- **{key}:** {value}")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("✅ Approve", key=f"approve_{rec['id']}", type="primary", use_container_width=True):
                st.session_state.approved.append(rec['id'])
                st.rerun()
        
        with col2:
            if st.button("❌ Reject", key=f"reject_{rec['id']}", use_container_width=True):
                st.session_state.rejected.append(rec['id'])
                st.rerun()
        
        st.markdown("---")

# Show completion message if all processed
if len(st.session_state.approved) + len(st.session_state.rejected) == len(recommendations):
    st.success("🎉 All recommendations have been processed!")
    
    st.markdown("### Summary")
    st.markdown(f"- ✅ Approved: {len(st.session_state.approved)}")
    st.markdown(f"- ❌ Rejected: {len(st.session_state.rejected)}")
    
    if st.button("Reset Queue", type="secondary"):
        st.session_state.approved = []
        st.session_state.rejected = []
        st.rerun()

# Show message if some processed
elif len(st.session_state.approved) + len(st.session_state.rejected) > 0:
    remaining = len(recommendations) - len(st.session_state.approved) - len(st.session_state.rejected)
    st.info(f"📊 {remaining} recommendation(s) remaining")

# Show approved/rejected items at the bottom
if st.session_state.approved or st.session_state.rejected:
    st.markdown("---")
    st.markdown("### Processed Items")
    
    if st.session_state.approved:
        with st.expander("✅ Approved Items", expanded=False):
            for rec_id in st.session_state.approved:
                rec = next(r for r in recommendations if r['id'] == rec_id)
                st.markdown(f"- {rec['icon']} {rec['title']}")
    
    if st.session_state.rejected:
        with st.expander("❌ Rejected Items", expanded=False):
            for rec_id in st.session_state.rejected:
                rec = next(r for r in recommendations if r['id'] == rec_id)
                st.markdown(f"- {rec['icon']} {rec['title']}")