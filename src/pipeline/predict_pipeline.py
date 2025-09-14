import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Tenure: float,
        PreferredLoginDevice: str,
        CityTier: int,
        WarehouseToHome: float,
        PreferredPaymentMode: str,
        Gender: str,
        HourSpendOnApp: float,
        NumberOfDeviceRegistered: int,
        PreferedOrderCat: str,
        SatisfactionScore: int,
        MaritalStatus: str,
        NumberOfAddress: int,
        Complain: int,
        OrderAmountHikeFromlastYear: float,
        CouponUsed: float,
        OrderCount: float,
        DaySinceLastOrder: float,
        CashbackAmount: int):

        self.Tenure = Tenure
        self.PreferredLoginDevice = PreferredLoginDevice
        self.CityTier = CityTier
        self.WarehouseToHome = WarehouseToHome
        self.PreferredPaymentMode = PreferredPaymentMode
        self.Gender = Gender
        self.HourSpendOnApp = HourSpendOnApp
        self.NumberOfDeviceRegistered = NumberOfDeviceRegistered
        self.PreferedOrderCat = PreferedOrderCat
        self.SatisfactionScore = SatisfactionScore
        self.MaritalStatus = MaritalStatus
        self.NumberOfAddress = NumberOfAddress
        self.Complain = Complain
        self.OrderAmountHikeFromlastYear = OrderAmountHikeFromlastYear
        self.CouponUsed = CouponUsed
        self.OrderCount = OrderCount
        self.DaySinceLastOrder = DaySinceLastOrder
        self.CashbackAmount = CashbackAmount

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Tenure": [self.Tenure],
                "PreferredLoginDevice": [self.PreferredLoginDevice],
                "CityTier": [self.CityTier],
                "WarehouseToHome": [self.WarehouseToHome],
                "PreferredPaymentMode": [self.PreferredPaymentMode],
                "Gender": [self.Gender],
                "HourSpendOnApp": [self.HourSpendOnApp],
                "NumberOfDeviceRegistered": [self.NumberOfDeviceRegistered],
                "PreferedOrderCat": [self.PreferedOrderCat],
                "SatisfactionScore": [self.SatisfactionScore],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfAddress": [self.NumberOfAddress],
                "Complain": [self.Complain],
                "OrderAmountHikeFromlastYear": [self.OrderAmountHikeFromlastYear],
                "CouponUsed": [self.CouponUsed],
                "OrderCount": [self.OrderCount],
                "DaySinceLastOrder": [self.DaySinceLastOrder],
                "CashbackAmount": [self.CashbackAmount],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)