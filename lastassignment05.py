# What I learned as a student while practicing this:
# 1. End-to-end ML pipeline: load → clean → engineer target → select features → split → train → evaluate.
# 2. Decision Trees are great for interpretability (we can actually see the rules!).
# 3. Always check class balance, use proper y as Series, and print everything useful.

import pandas as pd
from sklearn.metrics import accuracy, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

#  1. LOAD THE DATA
data = pd.read_csv('data/data_weather.csv')

print("=== Columns in the dataset ===")
print(data.columns.tolist())

print("\n=== First 5 rows of raw data ===")
print(data.head())

print("\n=== Rows containing any null values ===")
print(data[data.isnull().any(axis=1)])

#  2. DATA CLEANING
# Remove the unnecessary 'number' column (looks like a row index)
if 'number' in data.columns:
    del data['number']
    print("\n[number] column removed successfully.")

# Drop rows with missing values (NaNs)
before_rows = data.shape[0]
print(f"\nRows before dropping NaNs: {before_rows}")

data = data.dropna()

after_rows = data.shape[0]
print(f"Rows after dropping NaNs: {after_rows}")
print(f"Total rows dropped due to cleaning: {before_rows - after_rows}")

#  3. CONVERT TO CLASSIFICATION TASK
clean_data = data.copy()

# Create binary target: high_humidity_label = 1 if relative_humidity_3pm > 24.99
# (This threshold is specific to the tutorial dataset — it creates a good balance of 0/1 classes)
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99).astype(int)

# Check class balance (very important for classification!)
print("\n=== High Humidity Label Distribution ===")
print(clean_data['high_humidity_label'].value_counts())
print(f"Percentage of high humidity (1): {clean_data['high_humidity_label'].mean()*100:.2f}%")

# Target variable y (as Series — best practice for sklearn)
y = clean_data['high_humidity_label'].copy()

print("\n=== First 7 values of target y ===")
print(y.head(7))

# 4. FEATURE SELECTION 
# Use only 9am sensor readings to predict 3pm humidity (realistic morning forecast scenario)
morning_features = [
    'air_pressure_9am',
    'air_temp_9am',
    'avg_wind_direction_9am',
    'avg_wind_speed_9am',
    'max_wind_direction_9am',
    'max_wind_speed_9am',
    'rain_accumulation_9am',
    'rain_duration_9am'
]

X = clean_data[morning_features].copy()

print("\n=== Features used in X ===")
print(X.columns.tolist())
print(f"X shape: {X.shape}")

#  5. TRAIN-TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=324
)

print("\n=== Train/Test Split Summary ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

# Quick statistics on target in training set
print("\n=== Statistical summary of y_train ===")
print(y_train.describe())

# 6. TRAIN THE DECISION TREE CLASSIFIER 
humidity_classifier = DecisionTreeClassifier(
    max_leaf_nodes=10,      # Keeps the tree small and interpretable (prevents overfitting)
    random_state=0
)

humidity_classifier.fit(X_train, y_train)
print("\nModel trained successfully! (DecisionTreeClassifier with max_leaf_nodes=10)")

# 7. PREDICT ON TEST SET
predictions = humidity_classifier.predict(X_test)

print("\n=== Sample Predictions vs Actual (first 10) ===")
print("Predictions:", predictions[:10])
print("Actual     :", y_test[:10].values)

#  8. EVALUATE THE MODEL 
accuracy = accuracy(y_test, predictions)
print(f"\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# More detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 9. VISUALIZE THE DECISION TREE (RULES)
# This is the best part for learning — we can actually read the tree!
print("\n=== DECISION TREE RULES (Text Version) ===")
tree_rules = export_text(
    humidity_classifier,
    feature_names=morning_features,
    show_weights=True
)
print(tree_rules)

#  10. FEATURE IMPORTANCE 
print("\n=== Feature Importance (how much each morning feature contributes) ===")
importances = humidity_classifier.feature_importances_
for feature, importance in sorted(zip(morning_features, importances), key=lambda x: x[1], reverse=True):
    print(f"{feature:25} : {importance:.4f}")

print("\n=== Project Complete! ===")
print("Key learning: Decision Trees are interpretable and work well on tabular weather data.")
print("Next steps you can try as a student:")
print("   • Change max_leaf_nodes and see how accuracy changes")
print("   • Try max_depth instead of max_leaf_nodes")
print("   • Add cross-validation with cross_val_score")
print("   • Plot the actual tree using plot_tree + matplotlib")