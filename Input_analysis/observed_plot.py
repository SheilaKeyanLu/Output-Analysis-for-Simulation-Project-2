import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Get current script directory
dir_path = os.path.dirname(os.path.abspath(__file__))

# Define data file path
data_path = os.path.join(dir_path, "data_generation", "Data_File.xlsx")

# Create timestamp for output folder
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory
output_dir = os.path.join(dir_path, "output", "eda_output", time_str)
os.makedirs(output_dir, exist_ok=True)

# Load Excel file
df = pd.read_excel(data_path, header=None)

# Labels corresponding to each row
labels = [
    "Interarrival Times",
    "Service Times for Initial Phase",
    "Service Times for Placing Keyboard and Mouse",
    "Service Times for Assembling Case (Aluminum Plates)"
]

# Iterate through each row
for i in range(len(labels)):
    row = df.iloc[i]

    # Remove first column (label) and drop missing values
    data = row.iloc[1:].dropna().astype(float)

    # Plot histogram
    plt.figure()
    plt.hist(data, bins=20, density=True)
    plt.title(labels[i])
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Save figure
    save_path = os.path.join(output_dir, f"{labels[i]}.png")
    plt.savefig(save_path)