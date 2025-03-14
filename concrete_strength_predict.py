
import joblib
import pandas as pd


model = joblib.load("monotonic_model.pkl")

new_data = pd.read_csv("/Users/kuotingyu/Downloads/new_concrete_data/new_concrete_data.csv")

new_data["predicted_strength"] = model.predict(new_data)


print(new_data.head())