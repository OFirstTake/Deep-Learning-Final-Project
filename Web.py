import streamlit as st
import requests
import datetime
import pandas as pd

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

# Create 2 columns for the user to select the Company and Date
col1, col2 = st.columns(2)

with col1:
    selected_company = st.selectbox("Select a Company:", VN_technology, index=VN_technology.index("FPT"))

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
        "company": selected_company,
        "date": selected_date.strftime("%Y-%m-%d")
    }

    with st.spinner(f'Analyzing data and history for {selected_company}...'):
        try:
            # Send POST request to the API
            response = requests.post(api_url, json=payload)
            
            # If the API returns a success status
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")
                
                # Visualize the 30-day historical Open Prices leading up to the target date
                st.subheader(f"30-Day Historical Open Prices ({selected_company})")
                
                # Create a DataFrame for the chart
                history_df = pd.DataFrame({
                    "Date": result["history_dates"],
                    "Open Price (VND)": result["history_prices"]
                })
                # Set Date as index so the chart x-axis formats correctly
                history_df.set_index("Date", inplace=True)
                
                # Render the line chart
                st.line_chart(history_df, use_container_width=True)
                
                st.markdown("---")
                
                # Display the predicted Open price for the target date
                m1, = st.columns(1)
                m1.metric(
                    label=f"Predicted Open Price for {result['target_date']}", 
                    value=f"{result['predicted_open_price_VND']:,.0f} VND"
                )
                
            else:
                # If the API returns an error
                error_data = response.json()
                st.error(f"Error: {error_data.get('detail', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API! Make sure you are running 'uvicorn main:app --reload' in another terminal.")