# streamlit_app.py
import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Customer Churn Project", layout="centered")
st.title("📊 Customer Churn Predictor")

st.subheader("Fill in the below details:")

Tenure = st.number_input("Enter the Tenure [0 - 30]", min_value=0, max_value=30, step=1)
PreferredLoginDevice = st.selectbox("Preferred Login Device", ["Select", "Computer", "Mobile Phone", "Phone"])
CityTier = st.number_input("Enter the City Tier [1 - 3]", min_value=1, max_value=3, step=1)
WarehouseToHome = st.number_input("Enter the Warehouse To Home distance [1 - 35]", min_value=1, max_value=35, step=1)
PreferredPaymentMode = st.selectbox("Preferred Payment Mode", ["Select", "CC", "COD", "Cash on Delivery",
    "Credit Card", "Debit Card", "E wallet", "UPI"])
Gender = st.selectbox("Gender", ["Select", "Female", "Male"])
HourSpendOnApp = st.number_input("Enter the Number of Hours spent on App [1 - 4]", min_value=1, max_value=4, step=1)
NumberOfDeviceRegistered = st.number_input("Enter the Number of Device Registered [1 - 5]", min_value=1, max_value=5, step=1)
PreferedOrderCat = st.selectbox("Preferred Order Category", ["Select", "Laptop & Accessory", "Mobile Phone",
    "Fashion", "Mobile", "Grocery", "Others"])
SatisfactionScore = st.number_input("Enter the Satisfaction Score [1 - 5]", min_value=1, max_value=5, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Select","Single", "Married", "Divorced"])
NumberOfAddress = st.number_input("Enter the Number Of Address Registered [1 - 12]", min_value=1, max_value=12, step=1)
Complain = st.number_input("Complain [0 - 1]", min_value=0, max_value=1, step=1)
OrderAmountHikeFromlastYear = st.number_input("Enter the Order Amount Hike From last Year [1 - 25]", min_value=1, max_value=25, step=1)
CouponUsed = st.number_input("Enter the Number of Coupons Used [1 - 3]", min_value=1, max_value=3, step=1)
OrderCount = st.number_input("Enter the Order Count [1 - 6]", min_value=1, max_value=6, step=1)
DaySinceLastOrder = st.number_input("Enter the Days Since Last Ordered [1 - 15]", min_value=1, max_value=15, step=1)
CashbackAmount = st.number_input("Enter the Cashback Amount Received [1 - 270]", min_value=0, max_value=270, step=1)

if st.button("Predict Customer Churn"):
    if "Select" in (PreferredLoginDevice, PreferredPaymentMode, Gender,
                    PreferedOrderCat, MaritalStatus):
        st.warning("Please fill all the required fields.")
    else:
        data = CustomData(
            Tenure=Tenure,
            PreferredLoginDevice=PreferredLoginDevice,
            CityTier=CityTier,
            WarehouseToHome=WarehouseToHome,
            PreferredPaymentMode=PreferredPaymentMode,
            Gender=Gender,
            HourSpendOnApp=HourSpendOnApp,
            NumberOfDeviceRegistered=NumberOfDeviceRegistered,
            PreferedOrderCat=PreferedOrderCat,
            SatisfactionScore=SatisfactionScore,
            MaritalStatus=MaritalStatus,
            NumberOfAddress=NumberOfAddress,
            Complain=Complain,
            OrderAmountHikeFromlastYear=OrderAmountHikeFromlastYear,
            CouponUsed=CouponUsed,
            OrderCount=OrderCount,
            DaySinceLastOrder=DaySinceLastOrder,
            CashbackAmount=CashbackAmount
        )
        df = data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        st.success(f"🎯 Customer Churn Prediction: {prediction[0]}")