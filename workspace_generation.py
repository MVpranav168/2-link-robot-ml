import mujoco
import numpy as np
import matplotlib.pyplot as plt

# MuJoCo model
xml = """
<mujoco model="two_link_arm">

    <worldbody>

        <!-- Link 1 -->
        <body name="link1" pos="0 0 0.05">

            <!-- Joint 1 -->
            <joint
                name="joint1"
                type="hinge"
                pos="0 0 0"
                axis="0 0 1"
            />

            <!-- Link 1: 30 cm -->
            <geom
                name="link1_geom"
                type="capsule"
                fromto="0 0 0 0.30 0 0"
                size="0.025"
            />

            <!-- Link 2 -->
            <body name="link2" pos="0.30 0 0">

                <!-- Joint 2 -->
                <joint
                    name="joint2"
                    type="hinge"
                    pos="0 0 0"
                    axis="0 0 1"
                />

                <!-- Link 2: 15 cm -->
                <geom
                    name="link2_geom"
                    type="capsule"
                    fromto="0 0 0 0.15 0 0"
                    size="0.022"
                />

                <!-- End effector -->
                <site
                    name="end_effector"
                    pos="0.15 0 0"
                    size="0.03"
                />

            </body>

        </body>

    </worldbody>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("MuJoCo workspace model loaded successfully!")
print("Number of joints:", model.njnt)

# Number of random configurations
n_samples = 5000

# Reproducible random seed
np.random.seed(42)

# Generate random joint angles in radians
q1_values = np.random.uniform(-np.pi, np.pi, n_samples)
q2_values = np.random.uniform(-np.pi, np.pi, n_samples)

print("Number of configurations:", n_samples)
print("q1 range: -180° to +180°")
print("q2 range: -180° to +180°")

# Arrays to store end-effector positions
x_positions = np.zeros(n_samples)
y_positions = np.zeros(n_samples)

# Simulate each joint configuration
for i in range(n_samples):

    # Set joint angles
    data.qpos[0] = q1_values[i]
    data.qpos[1] = q2_values[i]

    # Update MuJoCo kinematics
    mujoco.mj_forward(model, data)

    # Read end-effector position
    x_positions[i] = data.site_xpos[0][0]
    y_positions[i] = data.site_xpos[0][1]

print("Simulation completed successfully!")
print("Collected", len(x_positions), "end-effector positions.")

# Plot the simulated workspace
plt.figure(figsize=(8, 8))

plt.scatter(
    x_positions,
    y_positions,
    s=5,
    alpha=0.5
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("2-Link Robot Workspace — MuJoCo Simulation")

plt.axis("equal")
plt.grid(True)

plt.show()

# Create a DataFrame containing the simulated dataset
import pandas as pd

workspace_df = pd.DataFrame({
    "q1_rad": q1_values,
    "q2_rad": q2_values,
    "x_m": x_positions,
    "y_m": y_positions
})

# Save the dataset
workspace_df.to_csv("mujoco_workspace_dataset.csv", index=False)

print("\nWorkspace dataset saved as:")
print("mujoco_workspace_dataset.csv")

print("\nFirst 5 samples:")
print(workspace_df.head())

print("\nWorkspace limits:")
print("Maximum X:", x_positions.max(), "m")
print("Minimum X:", x_positions.min(), "m")
print("Maximum Y:", y_positions.max(), "m")
print("Minimum Y:", y_positions.min(), "m")
