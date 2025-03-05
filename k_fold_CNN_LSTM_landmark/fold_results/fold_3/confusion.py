import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example confusion matrices (replace with your own)
cnn_cm = np.array([[20, 2, 0, 1,0,0,0], [0, 8, 0, 0,0,0,0], [0, 0, 16, 0,0,0,0], [1, 1, 0, 13,0,0,0], [0, 0, 0, 0,10,0,0], [0, 0, 0, 0,0,23,0], [0, 0, 0, 0,0,0,7]])  # CNN confusion matrix
cnn_lstm_cm = np.array([[23, 1, 0, 0,0,0,0], [1, 6, 0, 0,0,0,0], [0, 0, 18, 0,1,0,0], [0, 1, 0, 12,0,0,1], [0, 0, 0, 0,11,0,0], [0, 0, 0, 0,0,22,0], [0, 0, 0, 0,0,0,7]])  # CNN-LSTM confusion matrix

# Plot CNN confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press'],
            yticklabels=['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press'])
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("cnn_confusion_matrix.png")

# Plot CNN-LSTM confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cnn_lstm_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press'],
            yticklabels=['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press'])
plt.title("CNN-LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("cnn_lstm_confusion_matrix.png")
