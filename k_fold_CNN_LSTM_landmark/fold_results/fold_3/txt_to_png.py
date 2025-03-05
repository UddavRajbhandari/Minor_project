import matplotlib.pyplot as plt

# Read classification report text
with open("classification_report.txt", "r") as file:
    report_text = file.read()

# Create figure
plt.figure(figsize=(8, 4))
plt.text(0, 1, report_text, fontsize=10, family="monospace", verticalalignment="top")
plt.axis("off")
plt.savefig("classification_report.png", bbox_inches="tight", dpi=300)
plt.show()
