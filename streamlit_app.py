import os
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# API Base URL configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

# Set page config
st.set_page_config(
    page_title="Review Router AI - Support Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .badge-positive {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-neutral {
        background-color: #9e6a03;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-negative {
        background-color: #da3633;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-urgent {
        background-color: #bd561d;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-high {
        background-color: #d97706;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-medium {
        background-color: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-low {
        background-color: #4b5563;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .issue-box {
        background: rgba(22, 27, 34, 0.8);
        border-left: 4px solid #58a6ff;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to call FastAPI backend with direct python fallback
@st.cache_resource
def load_fallback_services():
    from app.services.sentiment import SentimentAnalyzer
    from app.services.aspect_extraction import AspectExtractor
    from app.services.routing import SupportRouter
    from app.services.ticket_service import TicketService
    from app.db.database import SessionLocal, Base, engine
    
    Base.metadata.create_all(bind=engine)
    sentiment_analyzer = SentimentAnalyzer()
    aspect_extractor = AspectExtractor(sentiment_analyzer=sentiment_analyzer)
    support_router = SupportRouter()
    ticket_service = TicketService(sentiment_analyzer, aspect_extractor, support_router)
    return ticket_service, SessionLocal

def analyze_review_api(text: str, cust_id: str, prod_id: str):
    # Try calling FastAPI REST API first
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"review_text": text, "customer_id": cust_id, "product_id": prod_id},
            timeout=5
        )
        if response.status_code == 200:
            return response.json(), "fastapi_http"
    except Exception:
        pass
        
    # Fallback to direct python service call
    ticket_service, _ = load_fallback_services()
    res = ticket_service.analyze_and_route(text, cust_id, prod_id)
    return res, "direct_python"

def create_ticket_api(text: str, cust_id: str, prod_id: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/tickets",
            json={"review_text": text, "customer_id": cust_id, "product_id": prod_id},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
        
    ticket_service, SessionLocal = load_fallback_services()
    res = ticket_service.analyze_and_route(text, cust_id, prod_id)
    db = SessionLocal()
    try:
        from app.db.models import SupportTicket
        summary_txt = f"{res['overall_priority']} Issue for {res['primary_department']}: " + "; ".join([i['opinion'] for i in res['issues']])
        ticket = SupportTicket(
            ticket_id=res["ticket_id"],
            customer_id=res["customer_id"],
            product_id=res["product_id"],
            department=res["primary_department"],
            priority=res["overall_priority"],
            status="OPEN",
            review_text=res["review_text"],
            summary=summary_txt[:300]
        )
        db.add(ticket)
        db.commit()
        return {"ticket_id": res["ticket_id"]}
    finally:
        db.close()

# Sidebar
st.sidebar.title("⚡ Review Router AI")
st.sidebar.caption("Intelligent E-Commerce Customer Review Analyzer & Router")

nav_choice = st.sidebar.radio(
    "Navigation Desk",
    ["🔍 Live Review Analyzer", "📁 Batch Review Inspector", "🎫 Support Ticket Desk", "📊 Analytics & Insights"]
)

# Test Backend API Connectivity
api_online = False
try:
    r = requests.get(API_BASE_URL.replace("/api/v1", "/"), timeout=2)
    api_online = (r.status_code == 200)
except Exception:
    api_online = False

st.sidebar.markdown("---")
if api_online:
    st.sidebar.success("🟢 FastAPI Backend Connected")
else:
    st.sidebar.warning("🟡 Standalone Mode (FastAPI Offline)")

st.sidebar.info("""
**System Architecture:**
- **Sentiment Model:** DistilBERT / TF-IDF
- **Aspect Engine:** Rule & Context Guided ATE
- **Department Routing:** Automated Rule Engine
- **Backend:** FastAPI & SQLAlchemy DB
""")

# TAB 1: LIVE REVIEW ANALYZER
if nav_choice == "🔍 Live Review Analyzer":
    st.title("🔍 Live Review Analyzer & Router")
    st.caption("Paste customer feedback to analyze multi-aspect sentiment, assign priority, and route to the appropriate support team.")
    
    col_presets, _ = st.columns([3, 1])
    with col_presets:
        preset_choice = st.selectbox(
            "Quick Demo Examples:",
            [
                "Custom Text Input",
                "The phone is excellent, but the delivery was very late and my refund has not arrived.",
                "Charged twice for my order! Payment failed but money was deducted immediately.",
                "Top notch quality! Arrived earlier than expected and packaging was pristine.",
                "Support agent was disconnected and useless when I asked about return pickup."
            ]
        )
    
    default_text = "" if preset_choice == "Custom Text Input" else preset_choice
    
    review_input = st.text_area(
        "Customer Review Text:",
        value=default_text,
        height=130,
        placeholder="Enter customer review here..."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        cust_id = st.text_input("Customer ID", value="CUST-98214")
    with col2:
        prod_id = st.text_input("Product ID", value="PROD-PHONE-X")
        
    if st.button("🚀 Analyze & Route Review", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a valid review text.")
        else:
            with st.spinner("Processing NLP models & routing logic..."):
                res, mode = analyze_review_api(review_input, cust_id, prod_id)
                
            st.markdown("---")
            st.subheader("📋 Analysis & Routing Summary")
            
            m1, m2, m3, m4 = st.columns(4)
            
            # Sentiment Badge
            sent = res["overall_sentiment"]
            sent_color = "badge-positive" if sent == "Positive" else ("badge-negative" if sent == "Negative" else "badge-neutral")
            m1.markdown(f"**Overall Sentiment**<br><span class='{sent_color}'>{sent}</span> ({res['overall_confidence']*100:.1f}%)", unsafe_allow_html=True)
            
            # Department Badge
            m2.markdown(f"**Primary Department**<br>📌 **{res['primary_department']}**", unsafe_allow_html=True)
            
            # Priority Badge
            prio = res["overall_priority"]
            prio_color = f"badge-{prio.lower()}"
            m3.markdown(f"**Calculated Priority**<br><span class='{prio_color}'>{prio}</span>", unsafe_allow_html=True)
            
            # Model Engine Badge
            engine_label = f"DISTILBERT ({'HTTP API' if mode == 'fastapi_http' else 'LOCAL'})" if res.get('model_type') == 'transformer' else f"{res.get('model_type', 'baseline').upper()} ({'HTTP API' if mode == 'fastapi_http' else 'LOCAL'})"
            m4.markdown(f"**NLP Engine**<br>🤖 `{engine_label}`", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Aspect & Opinion Breakdown
            st.markdown("### 🧩 Detected Aspects & Opinion Clauses")
            for issue in res["issues"]:
                st.markdown(f"""
                <div class='issue-box'>
                    <strong>Aspect:</strong> <code>{issue['aspect'].upper()}</code> &nbsp;|&nbsp;
                    <strong>Department:</strong> {issue['department']} &nbsp;|&nbsp;
                    <strong>Sentiment:</strong> <b>{issue['sentiment']}</b> ({issue['confidence']*100:.0f}% conf)<br>
                    <em>"{issue['opinion']}"</em>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 🔀 Routing Matrix")
            df_route = pd.DataFrame(res["routing"])
            st.dataframe(df_route, width='stretch')
            
            # Create Ticket Option
            st.markdown("---")
            if st.button("➕ Create Support Ticket in Database"):
                t_res = create_ticket_api(review_input, cust_id, prod_id)
                st.success(f"Support Ticket `{t_res.get('ticket_id', res['ticket_id'])}` successfully saved to Database & dispatched to {res['primary_department']}!")
                    
            with st.expander("🛠️ Raw JSON Response Payload"):
                st.json(res)

# TAB 2: BATCH REVIEW INSPECTOR
elif nav_choice == "📁 Batch Review Inspector":
    st.title("📁 Batch Review Inspector")
    st.caption("Upload a dataset CSV or process sample dataset to evaluate bulk routing performance.")
    
    data_file = st.file_uploader("Upload Reviews CSV (Must contain 'review' column)", type=["csv"])
    
    if data_file is not None:
        batch_df = pd.read_csv(data_file)
    else:
        st.info("No CSV uploaded. Showing sample records from dataset.")
        sample_path = "data/processed/ecommerce_reviews.csv"
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path).head(50)
        else:
            batch_df = pd.DataFrame()
            
    if not batch_df.empty:
        st.write(f"Loaded **{len(batch_df)}** reviews for batch processing.")
        st.dataframe(batch_df[['review_id', 'review', 'sentiment', 'priority']].head(10) if 'review_id' in batch_df.columns else batch_df.head(10), width='stretch')
        
        if st.button("⚡ Run Batch Analysis"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in batch_df.iterrows():
                rev = str(row['review'])
                res, _ = analyze_review_api(rev, "CUST-BATCH", "PROD-BATCH")
                results.append({
                    "review_text": rev,
                    "predicted_sentiment": res["overall_sentiment"],
                    "confidence": res["overall_confidence"],
                    "primary_department": res["primary_department"],
                    "overall_priority": res["overall_priority"],
                    "detected_aspects": ", ".join([i['aspect'] for i in res['issues']])
                })
                progress_bar.progress((idx + 1) / len(batch_df))
                
            res_df = pd.DataFrame(results)
            st.success("Batch analysis complete!")
            st.dataframe(res_df, width='stretch')
            
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Routing Report (CSV)",
                data=csv_data,
                file_name="batch_review_routing_results.csv",
                mime="text/csv"
            )

# TAB 3: SUPPORT TICKET DESK
elif nav_choice == "🎫 Support Ticket Desk":
    st.title("🎫 Support Ticket Desk Queue")
    st.caption("Manage, filter, and resolve routed customer support tickets stored in SQLite database.")
    
    # Try fetching tickets from FastAPI REST API first
    tickets_data = []
    try:
        r = requests.get(f"{API_BASE_URL}/tickets", timeout=3)
        if r.status_code == 200:
            tickets_data = r.json()
    except Exception:
        pass
        
    if not tickets_data:
        _, SessionLocal = load_fallback_services()
        db = SessionLocal()
        try:
            from app.db.models import SupportTicket
            db_tickets = db.query(SupportTicket).order_by(SupportTicket.created_at.desc()).all()
            tickets_data = [
                {
                    "ticket_id": t.ticket_id,
                    "customer_id": t.customer_id,
                    "department": t.department,
                    "priority": t.priority,
                    "status": t.status,
                    "summary": t.summary,
                    "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else ""
                }
                for t in db_tickets
            ]
        finally:
            db.close()

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        dept_filter = st.selectbox("Department Filter", ["All", "Logistics", "Product Support", "Payments", "Refunds", "Returns / Replacement", "Customer Support"])
    with f_col2:
        prio_filter = st.selectbox("Priority Filter", ["All", "LOW", "MEDIUM", "HIGH", "URGENT"])
    with f_col3:
        status_filter = st.selectbox("Status Filter", ["All", "OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"])
        
    filtered_tickets = tickets_data
    if dept_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get("department") == dept_filter]
    if prio_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get("priority") == prio_filter]
    if status_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get("status") == status_filter]
        
    st.write(f"Total Tickets Matching Filters: **{len(filtered_tickets)}**")
    
    if filtered_tickets:
        formatted_list = []
        for t in filtered_tickets:
            formatted_list.append({
                "Ticket ID": t.get("ticket_id"),
                "Customer": t.get("customer_id"),
                "Department": t.get("department"),
                "Priority": t.get("priority"),
                "Status": t.get("status"),
                "Summary": t.get("summary"),
                "Created At": t.get("created_at", "")
            })
        st.dataframe(pd.DataFrame(formatted_list), width='stretch')
        
        st.markdown("---")
        st.subheader("✏️ Quick Update Ticket Status")
        u_col1, u_col2, u_col3 = st.columns([2, 2, 1])
        
        with u_col1:
            t_ids = [t.get("ticket_id") for t in filtered_tickets]
            selected_t_id = st.selectbox("Select Ticket ID", t_ids)
        with u_col2:
            new_status = st.selectbox("New Status", ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"])
        with u_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Update Status"):
                updated = False
                try:
                    r = requests.patch(f"{API_BASE_URL}/tickets/{selected_t_id}", json={"status": new_status}, timeout=3)
                    if r.status_code == 200:
                        updated = True
                except Exception:
                    pass
                    
                if not updated:
                    _, SessionLocal = load_fallback_services()
                    db = SessionLocal()
                    try:
                        from app.db.models import SupportTicket
                        t_obj = db.query(SupportTicket).filter(SupportTicket.ticket_id == selected_t_id).first()
                        if t_obj:
                            t_obj.status = new_status
                            db.commit()
                            updated = True
                    finally:
                        db.close()
                        
                if updated:
                    st.success(f"Ticket {selected_t_id} updated to {new_status}")
                    st.rerun()
    else:
        st.info("No tickets found matching current filters.")

# TAB 4: ANALYTICS & INSIGHTS
elif nav_choice == "📊 Analytics & Insights":
    st.title("📊 Analytics & Department Insights")
    st.caption("Aggregated analytics on customer review sentiment distribution and department workload.")
    
    analytics_data = {}
    try:
        r = requests.get(f"{API_BASE_URL}/analytics", timeout=3)
        if r.status_code == 200:
            analytics_data = r.json()
    except Exception:
        pass
        
    if analytics_data:
        total_revs = analytics_data.get("total_reviews_analyzed", 0)
        total_ticks = analytics_data.get("total_tickets_created", 0)
    else:
        _, SessionLocal = load_fallback_services()
        db = SessionLocal()
        try:
            from app.db.models import SupportTicket, ReviewAnalysis
            total_revs = db.query(ReviewAnalysis).count()
            total_ticks = db.query(SupportTicket).count()
        finally:
            db.close()
            
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Reviews Processed", total_revs)
    k2.metric("Total Tickets Created", total_ticks)
    k3.metric("Ticket Conversion Rate", f"{(total_ticks/total_revs*100):.1f}%" if total_revs > 0 else "0%")
    
    st.markdown("---")
    
    # Load processed CSV for visualizations
    sample_path = "data/processed/ecommerce_reviews.csv"
    if os.path.exists(sample_path):
        df_analytics = pd.read_csv(sample_path)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Overall Sentiment Breakdown")
            st.bar_chart(df_analytics['sentiment'].value_counts())
        with c2:
            st.subheader("Department Workload Share")
            st.bar_chart(df_analytics['department'].value_counts())
            
        st.subheader("Priority Distribution by Department")
        ct = pd.crosstab(df_analytics['department'], df_analytics['priority'])
        st.dataframe(ct, width='stretch')
    else:
        st.info("Dataset analytics file not found.")
