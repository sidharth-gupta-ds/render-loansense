import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import base64

# Configuration
API_BASE_URL = "https://render-loansense.onrender.com/api/v1"

# Page config
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #ffffff;
        font-weight: bold;
    }
    .approved {
        background-color: #2d5016;
        border: 2px solid #4caf50;
    }
    .rejected {
        background-color: #5d1a1a;
        border: 2px solid #f44336;
    }
    .recommendation-box {
        background-color: #3d4f7d;
        border: 2px solid #5c7cfa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #ffffff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def call_api(endpoint, data=None, method="POST"):
    """Make API calls to FastAPI backend"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_prediction(prediction_data):
    """Display prediction results"""
    # Handle both prediction and explanation response formats
    prediction = prediction_data.get('prediction', 'Unknown')
    probability = prediction_data.get('probability', 0)
    confidence = prediction_data.get('confidence', 'Unknown')
    
    if prediction.strip() == 'Approved':
        st.markdown(f"""
        <div class="prediction-box approved">
            <h3>✅ Loan Approved!</h3>
            <p><strong>Probability:</strong> {probability:.2%}</p>
            {f"<p><strong>Confidence:</strong> {confidence}</p>" if confidence != 'Unknown' else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box rejected">
            <h3>❌ Loan Rejected</h3>
            <p><strong>Probability:</strong> {probability:.2%}</p>
            {f"<p><strong>Confidence:</strong> {confidence}</p>" if confidence != 'Unknown' else ""}
        </div>
        """, unsafe_allow_html=True)

def create_shap_plot(shap_values, feature_names):
    """Create SHAP waterfall plot"""
    # Sort by absolute SHAP value
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create waterfall plot
    fig = go.Figure()
    
    # Use more vibrant colors that work well on dark backgrounds
    colors = ['#ff6b6b' if v < 0 else '#51cf66' for v in values]
    
    fig.add_trace(go.Bar(
        x=features,
        y=values,
        marker_color=colors,
        name='SHAP Values'
    ))
    
    fig.update_layout(
        title='Feature Impact on Loan Decision (SHAP Values)',
        xaxis_title='Features',
        yaxis_title='SHAP Value',
        xaxis_tickangle=-45,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_recommendations(recommendations):
    """Display recommendations"""
    st.subheader("💡 Recommendations for Improvement")
    
    for rec in recommendations:
        if rec.get('priority'):
            priority_color = {
                'High': '🔴',
                'Medium': '🟡',
                'Low': '🟢'
            }.get(rec['priority'], '⚪')
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>{priority_color} {rec['priority']} Priority</h4>
                <p><strong>Feature:</strong> {rec.get('feature', 'General')}</p>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                {f"<p><strong>Current Value:</strong> {rec['current_value']}</p>" if 'current_value' in rec else ""}
                <p><strong>Actionable:</strong> {'✅ Yes' if rec.get('actionable') else '❌ No'}</p>
            </div>
            """, unsafe_allow_html=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">🏦 Loan Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Prediction", "Batch Prediction", "Upload CSV", "API Status"]
    )
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Upload CSV":
        csv_upload_page()
    elif page == "API Status":
        api_status_page()

def single_prediction_page():
    """Single prediction page"""
    st.header("📋 Single Loan Application")
    
    # Input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=8000000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=25000000)
            loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=15)
        
        with col2:
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
            residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=5000000)
            commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=3000000)
            luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=2000000)
            bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=1000000)
        
        submitted = st.form_submit_button("Submit Application")
    
    if submitted:
        # Prepare data (add leading space to match encoder format)
        loan_data = {
            "no_of_dependents": no_of_dependents,
            "education": f" {education}",  # Add leading space
            "self_employed": f" {self_employed}",  # Add leading space
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Prediction", "Explanation", "Recommendations"])
        
        with tab1:
            st.subheader("🔮 Prediction Result")
            prediction_result = call_api("predict", loan_data)
            
            if prediction_result:
                display_prediction(prediction_result)
        
        with tab2:
            st.subheader("📊 SHAP Explanation")
            explanation_result = call_api("explain", loan_data)
            
            if explanation_result:
                display_prediction(explanation_result)
                
                # SHAP plot
                shap_fig = create_shap_plot(
                    explanation_result['shap_values'],
                    list(explanation_result['shap_values'].keys())
                )
                st.plotly_chart(shap_fig, use_container_width=True)
                
                # Top features
                st.subheader("🔍 Top Contributing Features")
                for feature in explanation_result['top_contributing_features']:
                    impact_color = "🟢" if feature['impact'] == 'Positive' else "🔴"
                    st.write(f"{impact_color} **{feature['feature']}**: {feature['shap_value']:.3f} (Rank: {feature['rank']})")
                
                # Feature impact
                if explanation_result['feature_impact']:
                    st.subheader("📈 Feature Impact Details")
                    for feature, impact in explanation_result['feature_impact'].items():
                        st.write(f"• **{feature}**: {impact}")
        
        with tab3:
            st.subheader("💡 Recommendations")
            recommendation_result = call_api("recommend", loan_data)
            
            if recommendation_result:
                current_prediction = recommendation_result.get('current_prediction', '').strip()
                if current_prediction == 'Approved':
                    st.success("🎉 Congratulations! Your loan is already approved!")
                else:
                    display_recommendations(recommendation_result['recommendations'])
                    
                    # Potential improvements
                    if recommendation_result.get('potential_improvements'):
                        st.subheader("📈 Potential Improvements")
                        for feature, improvement in recommendation_result['potential_improvements'].items():
                            st.write(f"• **{feature}**: +{improvement:.3f} probability increase")

def batch_prediction_page():
    """Batch prediction page"""
    st.header("📊 Batch Loan Predictions")
    
    # Download template
    st.subheader("📥 Download Template")
    if st.button("Download CSV Template"):
        template_response = requests.get(f"{API_BASE_URL}/template")
        if template_response.status_code == 200:
            st.download_button(
                label="Download Template CSV",
                data=template_response.content,
                file_name="loan_template.csv",
                mime="text/csv"
            )
    
    # Manual batch input
    st.subheader("✏️ Manual Batch Input")
    num_applications = st.number_input("Number of Applications", min_value=1, max_value=10, value=2)
    
    applications = []
    for i in range(num_applications):
        with st.expander(f"Application {i+1}"):
            col1, col2 = st.columns(2)
            
            with col1:
                no_of_dependents = st.number_input(f"Number of Dependents {i+1}", min_value=0, max_value=10, value=2, key=f"deps_{i}")
                education = st.selectbox(f"Education {i+1}", ["Graduate", "Not Graduate"], key=f"edu_{i}")
                self_employed = st.selectbox(f"Self Employed {i+1}", ["Yes", "No"], key=f"emp_{i}")
                income_annum = st.number_input(f"Annual Income {i+1} (₹)", min_value=0, value=8000000, key=f"income_{i}")
                loan_amount = st.number_input(f"Loan Amount {i+1} (₹)", min_value=0, value=25000000, key=f"loan_{i}")
                loan_term = st.number_input(f"Loan Term {i+1} (years)", min_value=1, max_value=30, value=15, key=f"term_{i}")
            
            with col2:
                cibil_score = st.number_input(f"CIBIL Score {i+1}", min_value=300, max_value=900, value=750, key=f"cibil_{i}")
                residential_assets_value = st.number_input(f"Residential Assets {i+1} (₹)", min_value=0, value=5000000, key=f"res_{i}")
                commercial_assets_value = st.number_input(f"Commercial Assets {i+1} (₹)", min_value=0, value=3000000, key=f"com_{i}")
                luxury_assets_value = st.number_input(f"Luxury Assets {i+1} (₹)", min_value=0, value=2000000, key=f"lux_{i}")
                bank_asset_value = st.number_input(f"Bank Assets {i+1} (₹)", min_value=0, value=1000000, key=f"bank_{i}")
            
            applications.append({
                "no_of_dependents": no_of_dependents,
                "education": f" {education}",  # Add leading space
                "self_employed": f" {self_employed}",  # Add leading space
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value
            })
    
    if st.button("Process Batch Applications"):
        batch_data = {"data": applications}
        batch_result = call_api("predict/batch", batch_data)
        
        if batch_result:
            st.subheader("📊 Batch Results")
            
            # Create results DataFrame
            results_df = pd.DataFrame([
                {
                    "Application": i+1,
                    "Prediction": pred.get('prediction', 'Unknown'),
                    "Probability": f"{pred.get('probability', 0):.2%}",
                    "Confidence": pred.get('confidence', 'Unknown')
                }
                for i, pred in enumerate(batch_result['predictions'])
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            approved_count = sum(1 for pred in batch_result['predictions'] if pred.get('prediction', '').strip() == 'Approved')
            st.metric("Approved Applications", f"{approved_count}/{len(batch_result['predictions'])}")

def csv_upload_page():
    """CSV upload page"""
    st.header("📁 Upload CSV for Batch Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display file contents
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 File Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process CSV"):
            # Send to API
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_BASE_URL}/predict/csv", files={"file": uploaded_file.getvalue()})
            
            if response.status_code == 200:
                st.success("✅ CSV processed successfully!")
                
                # Display download button
                st.download_button(
                    label="Download Results CSV",
                    data=response.content,
                    file_name="loan_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"❌ Error processing CSV: {response.text}")

def api_status_page():
    """API status page"""
    st.header("🔧 API Status & Information")
    
    # Health check
    health_status = call_api("health", method="GET")
    if health_status:
        st.success(f"✅ API Status: {health_status['status']}")
        st.info(f"Message: {health_status['message']}")
    else:
        st.error("❌ API is not responding")
    
    # Model information
    model_info = call_api("models", method="GET")
    if model_info:
        st.subheader("🤖 Model Information")
        st.json(model_info)
    
    # API endpoints
    st.subheader("📡 Available Endpoints")
    endpoints = [
        {"Endpoint": "/predict", "Method": "POST", "Description": "Single prediction"},
        {"Endpoint": "/predict/batch", "Method": "POST", "Description": "Batch predictions"},
        {"Endpoint": "/predict/csv", "Method": "POST", "Description": "CSV upload predictions"},
        {"Endpoint": "/explain", "Method": "POST", "Description": "SHAP explanations"},
        {"Endpoint": "/recommend", "Method": "POST", "Description": "Recommendations"},
        {"Endpoint": "/template", "Method": "GET", "Description": "Download CSV template"},
        {"Endpoint": "/health", "Method": "GET", "Description": "Health check"},
        {"Endpoint": "/models", "Method": "GET", "Description": "Model information"}
    ]
    
    st.dataframe(pd.DataFrame(endpoints), use_container_width=True)

if __name__ == "__main__":
    main()
