import streamlit as st
import requests
import datetime

# The Vietnamese companies list in technology
VN_technology = [
    "CKV", "CMG", "CMT", "ICT", "ELC", "FPT", "HIG", "HPT", "ITD", "KST", 
    "LTC", "ONE", "PMJ", "PMT", "POT", "SAM", "SBD", "SGT", "SMT", "SRA", 
    "SRB", "ST8", "TST", "UNI", "VAT", "VEC", "VIE", "VLA", "VTC", "VTE"
]

# Page configuration
st.set_page_config(page_title="VN Tech Stock Predictor", layout="centered")

st.title("Smart Stock Prediction")
st.markdown("Predicting the opening prices of Vietnamese technology stocks using a Deep Learning model.")
st.markdown("---")

# Create 2 columns for the user to select the Ticker and Date
col1, col2 = st.columns(2)

with col1:
    selected_ticker = st.selectbox("Select a Ticker:", VN_technology, index=VN_technology.index("FPT"))

with col2:
    # Set the default date
    default_date = datetime.date(2022, 5, 16)
    selected_date = st.date_input("Select Target Date:", default_date)

st.markdown("---")

# Predict button
if st.button("Predict Now", use_container_width=True):
    # The URL to the FastAPI service
    api_url = "http://127.0.0.1:8000/predict_stock"
    
    # Package the data to send
    payload = {
        "ticker": selected_ticker,
        "date": selected_date.strftime("%Y-%m-%d")
    }

    with st.spinner(f'Analyzing data for {selected_ticker}...'):
        try:
            # Send POST request to the API
            response = requests.post(api_url, json=payload)
            
            # If the API returns a success status
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")
                
                # Display the results using Streamlit's Metric components
                m1, m2, m3 = st.columns(3)
                
                m1.metric(
                    label="Predicted Price", 
                    value=f"{result['predicted_open_price_VND']:,.0f} VND"
                )
                
                m2.metric(
                    label="Actual Price", 
                    value=f"{result['actual_open_price_VND']:,.0f} VND"
                )
                
                # Get the difference to display the red/green delta arrow
                # Tính toán màu sắc: Đỏ nếu dự đoán lố (>0), Xanh nếu dự đoán thấp hơn (<0)
                diff = result['difference_VND']
                diff_color = "#ff4b4b" if diff > 0 else "#09ab3b" 
                diff_sign = "+" if diff > 0 else ""

                # Dùng HTML để ép màu cho số to
                m3.markdown(f"""
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <span style="font-size: 14px; color: rgb(250, 250, 250);">Difference</span>
                        <span style="font-size: 1.8rem; font-weight: 600; color: {diff_color};">
                            {diff_sign}{diff:,.0f} VND
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
            else:
                # If the API returns an error
                error_data = response.json()
                st.error(f"Error: {error_data.get('detail', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API! Make sure you are running 'uvicorn main:app --reload' in another terminal.")