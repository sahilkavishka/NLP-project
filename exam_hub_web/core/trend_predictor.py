import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression

class TrendPredictor:
    """
    Machine Learning module to predict 2026 exam trends based on past paper data.
    Uses Pandas for data manipulation and Scikit-Learn for Linear Regression.
    """
    def __init__(self):
        print("[TrendPredictor] Loading historical exam data...")
        # Simulated historical dataset (Frequency of questions per topic: 2016 - 2025)
        # We focus on advanced 2026 topics relevant to your university syllabus
        self.data = {
            "Year": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "AI_Driven_Data_Warehousing": [0, 0, 1, 2, 3, 5, 6, 8, 10, 12], # High Demand in 2026
            "Zero_Trust_Networks": [0, 1, 1, 2, 3, 4, 6, 7, 9, 11], # Trending
            "Basic_Subnetting": [10, 9, 9, 8, 7, 6, 5, 4, 3, 2], # Dropping (Outdated)
            "Generative_AI_in_Data_Science": [0, 0, 0, 0, 1, 2, 4, 7, 10, 14], # Very High Demand
            "Traditional_SQL": [8, 8, 7, 7, 6, 6, 5, 5, 4, 4] # Dropping
        }
        self.df = pd.DataFrame(self.data)
        print("[TrendPredictor] Data loaded successfully.")

    def predict_2026_trends(self):
        """
        Runs Linear Regression to predict the demand for each topic in 2026.
        """
        print("[TrendPredictor] Running Machine Learning Predictions for 2026...")
        predictions = {}
        X = self.df[['Year']] # Features

        for topic in self.df.columns:
            if topic == 'Year':
                continue
            
            y = self.df[topic] # Target
            
            # Train the ML Model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for 2026
            pred_2026 = model.predict([[2026]])[0]
            
            # Calculate if the trend is going Up or Down
            trend_direction = "Up 📈" if model.coef_[0] > 0 else "Down 📉"
            
            predictions[topic] = {
                "historical": list(y),
                "prediction_2026": max(0, round(pred_2026, 1)),
                "trend": trend_direction
            }
        
        return {
            "years": list(self.df['Year']),
            "predictions": predictions
        }