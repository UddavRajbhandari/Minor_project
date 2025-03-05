import matplotlib.pyplot as plt
import numpy as np

exercises = ['Barbell Biceps Curl', 'Hammer Curl', 'Lat Pulldown', 'Lateral Raise', 'Pull Up', 'Push-up', 'Shoulder Press']
cnn_precision =  [0.95, 0.73, 1.00, 0.93, 1.00, 1.00, 1.00]  # Replace with actual data
cnn_lstm_precision =[0.96, 0.75, 1.00, 1.00, 0.92, 1.00, 0.88]
cnn_recall =  [0.87, 1.00, 1.00, 0.87, 1.00, 1.00, 1.00]
cnn_lstm_recall = [0.96, 0.86, 0.95, 0.86, 1.00, 1.00, 1.00]


x = np.arange(len(exercises))  # Exercise categories
bar_width = 0.2  # Adjust width to prevent overlap

fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart for precision
ax.bar(x - 1.5 * bar_width, cnn_precision, bar_width, label='CNN Precision', color='blue')
ax.bar(x - 0.5 * bar_width, cnn_lstm_precision, bar_width, label='CNN-LSTM Precision', color='green')

# Bar chart for recall
ax.bar(x + 0.5 * bar_width, cnn_recall, bar_width, label='CNN Recall', color='lightblue', alpha=0.7)
ax.bar(x + 1.5 * bar_width, cnn_lstm_recall, bar_width, label='CNN-LSTM Recall', color='lightgreen', alpha=0.7)

# Adding labels and titles
ax.set_xlabel('Exercises')
ax.set_ylabel('Score')
ax.set_title('Precision and Recall Comparison: CNN vs. CNN-LSTM')
ax.set_xticks(x)
ax.set_xticklabels(exercises, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig("precision_recall_comparison.png")
plt.show()