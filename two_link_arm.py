import mujoco
import mujoco.viewer
import numpy as np

xml = """
<mujoco model="two_link_arm">

    <worldbody>

        <!-- Ground -->
        <geom
            name="ground"
            type="plane"
            size="2 2 0.01"
            pos="0 0 0"
            rgba="0.7 0.7 0.7 1"
        />

        <!-- Link 1 -->
        <body name="link1" pos="0 0 0.05">

            <!-- Joint 1 -->
            <joint
                name="joint1"
                type="hinge"
                pos="0 0 0"
                axis="0 0 1"
            />

            <!-- Link 1 geometry: 30 cm -->
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

                <!-- Link 2 geometry: 15 cm -->
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

# Create MuJoCo model
model = mujoco.MjModel.from_xml_string(xml)

# Create simulation data
data = mujoco.MjData(model)

print("2-Link Robot Model Loaded Successfully!")
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of position variables:", model.nq)

# Test multiple joint configurations
test_configurations = [
    (0, 0),
    (45, 30),
    (30, -45),
    (-60, 20),
    (90, -30)
]

print("\nTesting multiple joint configurations:")

for q1_deg, q2_deg in test_configurations:

    # Convert degrees to radians
    q1 = np.deg2rad(q1_deg)
    q2 = np.deg2rad(q2_deg)

    # Set MuJoCo joint positions
    data.qpos[0] = q1
    data.qpos[1] = q2

    # Update MuJoCo
    mujoco.mj_forward(model, data)

    # MuJoCo end-effector position
    ee_position = data.site_xpos[0]

    # Manual forward kinematics
    x_fk = 0.30 * np.cos(q1) + 0.15 * np.cos(q1 + q2)
    y_fk = 0.30 * np.sin(q1) + 0.15 * np.sin(q1 + q2)

    # Calculate errors
    x_error = abs(ee_position[0] - x_fk)
    y_error = abs(ee_position[1] - y_fk)

    print("\n-----------------------------")
    print("q1 =", q1_deg, "degrees")
    print("q2 =", q2_deg, "degrees")

    print("MuJoCo:")
    print("x =", ee_position[0], "m")
    print("y =", ee_position[1], "m")

    print("Manual FK:")
    print("x =", x_fk, "m")
    print("y =", y_fk, "m")

    print("Errors:")
    print("x error =", x_error, "m")
    print("y error =", y_error, "m")

#import time

#print("\nOpening MuJoCo viewer...")

#with mujoco.viewer.launch_passive(model, data) as viewer:
   # while viewer.is_running():
       # mujoco.mj_step(model, data)
       # viewer.sync()
       # time.sleep(0.01)
