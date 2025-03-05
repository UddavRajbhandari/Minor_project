import numpy as np
import matplotlib.pyplot as plt

# Sample data: Joint angles during a push-up (in degrees)
frames = np.arange(1, 101)  # 100 frames
elbow_angles = 90 + 30 * np.sin(2 * np.pi * frames / 100)  # Simulated elbow movement
shoulder_angles = 60 + 20 * np.sin(2 * np.pi * frames / 100 + np.pi / 6)

# Plot the angles
plt.figure(figsize=(8, 5))
plt.plot(frames, elbow_angles, label="Elbow Angle", color='b')
plt.plot(frames, shoulder_angles, label="Shoulder Angle", color='r', linestyle="--")

plt.xlabel("Frame Number")
plt.ylabel("Angle (degrees)")
plt.title("Joint Angles During Push-Up")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("joint_angles.png", dpi=300)
plt.show()
