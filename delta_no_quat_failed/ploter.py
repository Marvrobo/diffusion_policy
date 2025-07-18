import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "training_loss_log.csv"  # Make sure this path is correct
df = pd.read_csv(csv_path)

# Plot the training loss curve
plt.figure(figsize=(15, 9))
plt.plot(df['epoch'], df['avg_loss'], marker='o', linestyle='-', color='blue', label='Training Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as an image file (optional)
plt.savefig("training_loss_curve.png")

# Show the plot
plt.show()

