2-Link Robot Arm End-Effector Position Prediction using Machine Learning
A machine learning project that predicts the 2D end-effector position
(x, y) of a 2-link robotic arm from its two joint angles (q1, q2).
The project starts with a mathematically generated robotics dataset and
progressively evaluates different regression algorithms. It is designed
as a learning project connecting robot kinematics, supervised machine
learning, and robotics applications.
Project Overview
The robot is a planar 2-link arm with:
- Link 1 length: 30 cm
- Link 2 length: 15 cm
- Joint angle range: -π to +π radians
- Inputs: q1, q2
- Outputs: x, y end-effector position in centimeters
- Dataset size: 5,000 samples
The forward kinematics equations used to generate the ground-truth
positions are:
x = L1 cos(q1) + L2 cos(q1 + q2)

y = L1 sin(q1) + L2 sin(q1 + q2)
where:
L1 = 30 cm
L2 = 15 cm
The maximum theoretical reach of the arm is 45 cm.
Machine Learning Problem
The project treats robot end-effector position prediction as a
supervised regression problem:
Joint angles (q1, q2)
          ↓
   Machine Learning Model
          ↓
End-effector position (x, y)
Features
- q1_rad --- Joint 1 angle in radians
- q2_rad --- Joint 2 angle in radians
Targets
- x_cm --- End-effector X position in centimeters
- y_cm --- End-effector Y position in centimeters
Dataset Generation
The dataset was generated using Python and NumPy.
For each sample:
1. Random values for q1 and q2 were generated between -π and π.
2. Forward kinematics was applied.
3. The resulting x and y coordinates were stored as the target
   values.
A fixed random seed (42) was used so that the experiment is
reproducible.
Version 1 --- Random Forest Regression
Version 1 establishes the complete machine learning pipeline:
1. Define the robot parameters.
2. Generate 5,000 joint configurations.
3. Calculate end-effector positions using forward kinematics.
4. Create the dataset.
5. Split the dataset into:
   - 80% training --- 4,000 samples
   - 20% testing --- 1,000 samples
6. Train a RandomForestRegressor.
7. Predict the test-set end-effector positions.
8. Evaluate the predictions using MAE and R².
9. Visualize actual versus predicted positions.
Random Forest Configuration
n_estimators = 200
random_state = 42
n_jobs = -1
Version 1 Results
  Metric             Result
  X MAE           0.4003 cm
  Y MAE           0.4235 cm
  Overall MAE     0.4119 cm
  Overall MAE       4.12 mm
  X R²               0.9995
  Y R²               0.9993
  Overall R²         0.9994
The high R² is expected because the dataset is generated from
deterministic mathematical forward kinematics without real sensor,
mechanical, or measurement errors.
Version 2 --- Regression Model Comparison
Version 2 compares four regression approaches using the same dataset and
train/test split:
- Linear Regression
- Random Forest Regression
- Support Vector Regression (SVR) with an RBF kernel
- Multi-Layer Perceptron (MLP) Neural Network
Results
  Model                  MAE (cm)   MAE (mm)         R²
  Linear Regression       17.4430     174.43     0.2291
  Random Forest            0.4119       4.12     0.9994
  SVR (RBF)                0.0532      0.532   0.999993
  MLP Neural Network       0.2520       2.52     0.9997
These results show that the nonlinear models capture the trigonometric
relationship between joint angles and end-effector position much better
than simple linear regression on this synthetic dataset.
The results are specific to this dataset, train/test split, model
configurations, and experimental setup; they should not be interpreted
as a universal ranking of machine learning algorithms.
Evaluation Metrics
Mean Absolute Error (MAE)
MAE measures the average absolute difference between predicted and
actual values.
For example:
MAE = 0.4119 cm
means the average coordinate prediction error is approximately 4.12
mm.
R² Score
R² measures how much of the variance in the target values is explained
by the model.
An R² close to 1 indicates that the model explains most of the variation
in the target data.
For regression tasks, MAE and R² are used instead of classification
accuracy.
Visualization
The project includes visualizations comparing:
- Actual versus predicted end-effector positions
- Model MAE comparison
The actual and predicted position distributions from the Random Forest
model closely overlap because the prediction error is relatively small
compared with the robot's workspace.
Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- GitHub
Repository Structure
2-link-robot-ml/
│
├── 01_robot_arm_ml.ipynb
└── README.md
The notebook currently contains both Version 1 and Version 2
experiments.
Project Roadmap
Version 1 --- Completed
- 2-link robot kinematics
- Synthetic dataset generation
- Random Forest regression
- MAE and R² evaluation
- Actual vs predicted visualization
Version 2 --- Completed
- Linear Regression
- Random Forest
- SVR
- MLP Neural Network
- Model comparison
Version 3 --- Planned
Introduce simulated joint-sensor noise to investigate how prediction
performance changes when the input joint angles are no longer perfectly
measured.
Planned work includes:
- Simulated encoder/joint-angle noise
- Noisy training and/or testing inputs
- Comparison of model robustness
- MAE and R² analysis
Future Robotics Integration
The longer-term goal is to move from a purely mathematical dataset
toward a robotics workflow involving:
Robot / Simulator
      ↓
Joint measurements
      ↓
Machine Learning model
      ↓
Predicted end-effector position
Future versions may explore robot simulation and eventually real sensor
data.
Important Note
The current dataset is synthetic and generated from ideal forward
kinematics. Therefore, the reported machine learning errors are not
equivalent to the physical positioning accuracy of a real robotic arm.
A real robot would introduce factors such as:
- Encoder measurement error
- Mechanical backlash
- Joint tolerances
- Link-length manufacturing variation
- Structural flex
- Sensor noise
- Calibration errors
These effects will be considered in later versions of the project.
Author
MV Pranav
Robotics / Mechanical Engineering Student
License
This project is currently intended as a personal learning and portfolio
project.
