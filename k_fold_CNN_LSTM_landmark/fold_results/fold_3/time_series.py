import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load predictions from CSV
csv_path = r"D:\minor\Workout-Video-Classifier\src\u_deploy\predictions.csv"
df = pd.read_csv(csv_path)

# Ensure required columns exist
required_cols = {"frame", "cnn_prediction", "cnn_lstm_prediction"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_cols}")

# Extract data
frames = df["frame"].values
cnn_predictions = df["cnn_prediction"].astype("category").cat.codes  # Encode labels as numbers
cnn_lstm_predictions = df["cnn_lstm_prediction"].astype("category").cat.codes

# Get category mapping for labels
categories = df["cnn_prediction"].astype("category").cat.categories

# Plot
plt.figure(figsize=(12, 6))
plt.plot(frames, cnn_predictions, label="CNN Predictions", color='royalblue', linestyle='--', marker='o', markersize=5)
plt.plot(frames, cnn_lstm_predictions, label="CNN-LSTM Predictions", color='seagreen', linestyle='-', marker='x', markersize=5)

# Formatting
plt.xticks(rotation=45)
plt.yticks(ticks=range(len(categories)), labels=categories)  # Use actual labels
plt.xlabel("Frames")
plt.ylabel("Predicted Exercise Class")
plt.title("Model Predictions Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Save and Show
plt.savefig("model_predictions_over_time.png", dpi=300, bbox_inches="tight")
plt.show()
