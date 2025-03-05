import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load the image
img = mpimg.imread(r'D:\minor\k_fold_CNN_LSTM_landmark\fold_results\fold_3\training_curves.png')

# Define the splitting point (assuming a 50-50 split)
height, width, _ = img.shape
midpoint = width // 2  

# Split into accuracy and loss images
accuracy_img = img[:, :midpoint, :]
loss_img = img[:, midpoint:, :]

# Save the separated images with corrected paths
plt.imsave(r'D:\minor\k_fold_CNN_LSTM_landmark\fold_results\fold_3\accuracy_curve.png', accuracy_img)
plt.imsave(r'D:\minor\k_fold_CNN_LSTM_landmark\fold_results\fold_3\loss_curve.png', loss_img)

print("Accuracy and Loss curves have been saved as separate images.")
