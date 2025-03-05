# import numpy as np
# import matplotlib.pyplot as plt

# # Define the labels for the chart
# labels = ['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press']
# metrics = ['Precision', 'Recall', 'F1-Score']

# # Corrected data for CNN and CNN-LSTM (Replace these with your actual data)
# # Now each metric has 7 values, one for each exercise
# cnn_lstm_data = np.array([
#     [0.96, 0.75, 1.00, 1.00, 0.92, 1.00, 0.88],  # Precision
#     [0.96, 0.86, 0.95, 0.86, 1.00, 1.00, 1.00],  # Recall
#     [0.96, 0.80, 0.97, 0.92, 0.96, 1.00, 0.93],  # F1-Score
# ])

# cnn_data = np.array([
#     [0.95, 0.73, 1.00, 0.93, 1.00, 1.00, 1.00],  # Precision
#     [0.87, 1.00, 1.00, 0.87, 1.00, 1.00, 1.00],  # Recall
#     [0.91, 0.84, 1.00, 0.90, 1.00, 1.00, 1.00],  # F1-Score
# ])

# # Set up the radar chart
# angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
# angles += angles[:1]  # Add the first angle to the end to close the circle

# # Repeat the first value to close the circle for CNN and CNN-LSTM
# cnn_data = np.concatenate((cnn_data, cnn_data[:, [0]]), axis=1)
# cnn_lstm_data = np.concatenate((cnn_lstm_data, cnn_lstm_data[:, [0]]), axis=1)

# # Plotting CNN and CNN-LSTM
# fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# # Loop through the metrics (Precision, Recall, F1-Score)
# for i, metric in enumerate(metrics):
#     # Plot each metric for CNN
#     ax.plot(angles, cnn_data[i], linewidth=2, linestyle='solid', label=f'CNN {metric}')
#     ax.fill(angles, cnn_data[i], alpha=0.25)
    
#     # Plot each metric for CNN-LSTM
#     ax.plot(angles, cnn_lstm_data[i], linewidth=2, linestyle='solid', label=f'CNN-LSTM {metric}')
#     ax.fill(angles, cnn_lstm_data[i], alpha=0.25)

# # Set up the chart labels and title
# ax.set_yticklabels([])  # Hide radial axis labels
# ax.set_xticks(angles[:-1])  # Show category labels (exclude the last angle which is duplicate)
# ax.set_xticklabels(labels)

# ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# plt.title("Radar Chart Comparing CNN vs CNN-LSTM")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Metrics data for both models
metrics = ['Precision', 'Recall', 'F1-Score']
cnn_values = [0.96, 0.95, 0.95]  # Replace with actual values for CNN
cnn_lstm_values = [0.96, 0.95, 0.95]  # Replace with actual values for CNN-LSTM

# Number of metrics
num_metrics = len(metrics)

# Angle for each axis
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

# Complete the circle
cnn_values += cnn_values[:1]
cnn_lstm_values += cnn_lstm_values[:1]
angles += angles[:1]

# Plotting the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, cnn_values, color='blue', alpha=0.25, label='CNN')
ax.fill(angles, cnn_lstm_values, color='green', alpha=0.25, label='CNN-LSTM')

ax.set_yticklabels([])  # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title("CNN vs. CNN-LSTM: Model Comparison")
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("cnn_vs_cnnlstm_radar_chart.png")
