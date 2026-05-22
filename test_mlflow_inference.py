import mlflow
import pandas as pd
from pathlib import Path

# ชี้ไปยังโฟลเดอร์โปรเจคและฐานข้อมูล MLflow
PROJECT_ROOT = Path(__file__).resolve().parent
tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlruns.db'}"

print("1. Connecting to MLflow Database (mlruns.db)...")
mlflow.set_tracking_uri(tracking_uri)

print("2. Loading model 'SME_Promo_Recommender' (latest version) from Registry...")
model_name = "SME_Promo_Recommender"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

print(f"Model loaded successfully! Model type: {type(model)}")

print("\n3. Creating dummy input data (Customer 0, comparing with Item ID 10, 20, 30)...")
input_data = pd.DataFrame({
    "user_idx": [0, 0, 0],
    "item_idx": [10, 20, 30]
})
print(input_data.to_string(index=False))

print("\n4. Predicting interaction scores...")
predictions = model.predict(input_data)

# นำผลลัพธ์ที่ทำนายได้กลับไปใส่ในตาราง
input_data['predicted_score'] = predictions

print("\n================== Prediction Results ==================")
print(input_data.to_string(index=False))
print("======================================================")
