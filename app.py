import streamlit as st
import pandas as pd
import joblib
import numpy as np

# FIX 1: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="ShipmentSure: On-Time Delivery Predictor", layout="wide")

# --- 1. Load the Model and Scaler (Cached for Speed) ---
@st.cache_resource
def load_assets():
    """Loads the model and scaler only once."""
    try:
        # Load the trained XGBoost model
        model = joblib.load('xgboost_shipment_model.pkl')
        # Load the fitted StandardScaler
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Required asset file not found. Ensure 'xgboost_shipment_model.pkl' and 'scaler.pkl' are in the same folder as app.py. Details: {e}")
        st.stop()

# Load the assets
model, scaler = load_assets()

# --- 2. Preprocessing & Prediction Logic ---

def preprocess_and_predict(input_data, model, scaler):
    """Applies preprocessing and returns the prediction."""
    # Convert raw user inputs into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # FIX 2: DROP 'ID' because the model was trained WITHOUT IT.
    input_df = input_df.drop(columns=['ID']) 

    # 1. Feature Engineering: Cost_to_Weight_Ratio 
    input_df['Cost_to_Weight_Ratio'] = (
        input_df['Cost_of_the_Product'] / input_df['Weight_in_gms']
    )
    # Handle infinities/NaT from division by zero/missing values (Impute with median 0.055)
    input_df['Cost_to_Weight_Ratio'].replace([np.inf, -np.inf], 0.055, inplace=True) 
    input_df['Cost_to_Weight_Ratio'].fillna(0.055, inplace=True) 

    # 2. Encoding Categorical Features (Must match notebook)
    # Label Encoding
    # Assuming the mapping used in the notebook was {'low': 1, 'medium': 2, 'high': 0}
    pi_map = {'low': 1, 'medium': 2, 'high': 0} 
    input_df['Product_importance'] = input_df['Product_importance'].map(pi_map)

    gender_map = {'F': 0, 'M': 1}
    input_df['Gender'] = input_df['Gender'].map(gender_map)

    # One-Hot Encoding (drop_first=True)
    df_encoded = pd.get_dummies(input_df, columns=['Warehouse_block', 'Mode_of_Shipment'], drop_first=True)

    # 3. Define Expected Features (FIX 3: REMOVE 'ID' and ensure correct order)
    # Based on the feature list provided in your error message (the training data list)
    expected_features = [
        'Customer_care_calls', 
        'Customer_rating', 
        'Cost_of_the_Product', 
        'Prior_purchases', 
        'Product_importance', 
        'Gender', 
        'Discount_offered', 
        'Weight_in_gms', # End of original columns
        'Warehouse_block_B', 
        'Warehouse_block_C', 
        'Warehouse_block_D', 
        'Warehouse_block_F', 
        'Mode_of_Shipment_Road', 
        'Mode_of_Shipment_Ship', # End of OHE columns
        'Cost_to_Weight_Ratio' # Engineered feature (Must be last as per previous error hints)
    ]

    # Reindex the input DataFrame to match the model's expected column order (15 features)
    final_input = df_encoded.reindex(columns=expected_features, fill_value=0)
    
    # 4. Scaling Numerical Features (Applying loaded scaler)
    num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 
                'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
                
    # Scale numerical data
    final_input[num_cols] = scaler.transform(final_input[num_cols])

    # 5. Make Prediction
    prediction = model.predict(final_input)
    # Get probability for class 0 (Not Reached on Time)
    prediction_proba = model.predict_proba(final_input)[:, 0] 

    return prediction[0], prediction_proba[0]


# --- 3. Streamlit UI Design ---
st.title("📦 ShipmentSure: On-Time Delivery Predictor")
st.markdown("Use the controls below to predict the likelihood of a shipment **not reaching on time**.")
st.divider()

col1, col2, col3 = st.columns(3)
user_input = {}
# The input dictionary must still contain 'ID' so we can drop it later.
user_input['ID'] = 0 

with col1:
    st.header("Shipment Details")
    user_input['Warehouse_block'] = st.selectbox("Warehouse Block", ('A', 'B', 'C', 'D', 'F'))
    user_input['Mode_of_Shipment'] = st.selectbox("Mode of Shipment", ('Flight', 'Ship', 'Road'))
    user_input['Cost_of_the_Product'] = st.number_input("Cost of the Product ($)", min_value=92, max_value=310, value=150)
    user_input['Weight_in_gms'] = st.number_input("Weight (in gms)", min_value=100, max_value=7846, value=2500)

with col2:
    st.header("Customer & Product")
    user_input['Customer_care_calls'] = st.slider("Customer Care Calls", min_value=2, max_value=7, value=3)
    user_input['Prior_purchases'] = st.slider("Prior Purchases", min_value=2, max_value=10, value=3)
    user_input['Product_importance'] = st.selectbox("Product Importance", ('low', 'medium', 'high'))
    user_input['Gender'] = st.selectbox("Gender", ('F', 'M'))
    
with col3:
    st.header("Offer Details")
    user_input['Customer_rating'] = st.slider("Customer Rating (1=Worst, 5=Best)", min_value=1, max_value=5, value=4)
    user_input['Discount_offered'] = st.number_input("Discount Offered (%)", min_value=0, max_value=65, value=10)

st.divider()

if st.button("Predict Delivery Status", type="primary"):
    prediction, probability_not_on_time = preprocess_and_predict(user_input, model, scaler)
    
    st.subheader("Prediction Result")
    
    # Class 1 = Reached on Time; Class 0 = Not Reached on Time.
    if prediction == 1:
        st.success("✅ Prediction: Shipment **Will Reach On Time**.")
        st.metric(label="Likelihood of *Not Reaching on Time*", value=f"{probability_not_on_time:.2%}")
    else:
        st.error("❌ Prediction: Shipment **Will Not Reach On Time**.")
        st.metric(label="Likelihood of *Not Reaching on Time*", value=f"{probability_not_on_time:.2%}")
        st.info("Consider adjusting inputs (e.g., higher discount, different shipment mode) to reduce the risk of delay.")

st.caption("Reference: 'Reached.on.Time\_Y.N' = 1 (On Time), 0 (Not On Time).")