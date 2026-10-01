import pandas as pd
import numpy as np

# Load the MuJoCo workspace dataset
df = pd.read_csv("mujoco_workspace_dataset.csv")

# Calculate radial distance from the robot base
df["radius_m"] = np.sqrt(
    df["x_m"]**2 + df["y_m"]**2
)

# Theoretical workspace limits
theoretical_min = abs(0.30 - 0.15)
theoretical_max = 0.30 + 0.15

# Simulated workspace limits
simulated_min = df["radius_m"].min()
simulated_max = df["radius_m"].max()

print("=== Workspace Radial Validation ===")

print("\nTheoretical limits:")
print("Minimum radius:", theoretical_min, "m")
print("Maximum radius:", theoretical_max, "m")

print("\nSimulated limits:")
print("Minimum radius:", simulated_min, "m")
print("Maximum radius:", simulated_max, "m")

# Check whether every point lies inside the theoretical workspace
inside_workspace = (
    (df["radius_m"] >= theoretical_min) &
    (df["radius_m"] <= theoretical_max)
)

print("\nPoints inside theoretical workspace:",
      inside_workspace.sum(), "/", len(df))

print("Points outside theoretical workspace:",
      (~inside_workspace).sum())

# Final validation
if inside_workspace.all():
    print("\nVALIDATION PASSED")
    print("All 5000 MuJoCo points are inside the theoretical workspace.")
else:
    print("\nVALIDATION FAILED")
    print("Some points are outside the theoretical workspace.")
