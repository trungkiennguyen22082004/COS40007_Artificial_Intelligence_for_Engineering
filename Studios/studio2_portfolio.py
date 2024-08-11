import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ============================================================================================================================================================================================
# LOAD AND PLOT THE DATASET (WITH NO OUTLIERS)

# Load the dataset again for continuity
file_path = 'water_potability_no_outliers.csv'
water_df = pd.read_csv(file_path)

# Plotting the distribution of classes for Potability
plt.figure(figsize=(8, 6))
sns.countplot(x=water_df['Potability'], palette='viridis')
plt.title('Distribution of Potability Classes')
plt.xlabel('Potability')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not Potable (0)', 'Potable (1)'])
# plt.show()


# ============================================================================================================================================================================================
# DATASET WITH NORMALISED FEATURES

from sklearn.preprocessing import MinMaxScaler

normalised_water_df = water_df.copy()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# List of features to normalize (excluding Potability)
features_to_normalize = normalised_water_df.columns.drop('Potability')

# Normalize the features
normalised_water_df[features_to_normalize] = scaler.fit_transform(normalised_water_df[features_to_normalize])

# Save the normalized data to a new CSV file
normalised_water_df.to_csv("normalised_water_potability.csv", index=False)

print("==============================================================================================")
print("NORMALIZED DATA")
print(normalised_water_df)


# ============================================================================================================================================================================================
# DATASET WITH COMPOSITING FEATURES

import numpy as np

normalised_water_df_with_compositing_features = normalised_water_df.copy()

# Create composite features by calculating covariance
normalised_water_df_with_compositing_features['pH_Hardness_Cov'] = (normalised_water_df_with_compositing_features['ph'] - normalised_water_df_with_compositing_features['ph'].mean()) * (normalised_water_df_with_compositing_features['Hardness'] - normalised_water_df_with_compositing_features['Hardness'].mean())
normalised_water_df_with_compositing_features['Sulfate_Conductivity_Cov'] = (normalised_water_df_with_compositing_features['Sulfate'] - normalised_water_df_with_compositing_features['Sulfate'].mean()) * (normalised_water_df_with_compositing_features['Conductivity'] - normalised_water_df_with_compositing_features['Conductivity'].mean())
normalised_water_df_with_compositing_features['Turbidity_Organic_Carbon_Cov'] = (normalised_water_df_with_compositing_features['Turbidity'] - normalised_water_df_with_compositing_features['Turbidity'].mean()) * (normalised_water_df_with_compositing_features['Organic_carbon'] - normalised_water_df_with_compositing_features['Organic_carbon'].mean())
normalised_water_df_with_compositing_features['Chloramines_Trihalomethanes_Cov'] = (normalised_water_df_with_compositing_features['Chloramines'] - normalised_water_df_with_compositing_features['Chloramines'].mean()) * (normalised_water_df_with_compositing_features['Trihalomethanes'] - normalised_water_df_with_compositing_features['Trihalomethanes'].mean())

# Save the dataset with the new composite features
normalised_water_df_with_compositing_features.to_csv("normalised_water_potability_with_composites.csv", index=False)

print("==============================================================================================")
print("NORMALIZED DATA WITH NEW COMPOSITED FEATURES")
print(normalised_water_df_with_compositing_features)


# ============================================================================================================================================================================================
# DATASET WITH SELECTED NORMALISED FEATURES

# List of features to drop (weakest relationships)
features_to_drop = ['ph', 'Hardness', 'Sulfate', 'Turbidity']

# Create a new dataframe by dropping the selected features
selected_water_df = water_df.drop(columns=features_to_drop)

# Save the new dataframe to a CSV file
selected_water_df.to_csv("selected_features_water_potability.csv", index=False)

print("==============================================================================================")
print("ORIGINAL DATA WITH SELECTED FEATURES")
print(selected_water_df)


# ============================================================================================================================================================================================
# DATASET WITH SELECTED NORMALISED FEATURES

# List of features to drop (weakest relationships)
features_to_drop = ['ph', 'Hardness', 'Sulfate', 'Turbidity']

# Create a new dataframe by dropping the selected features
selected_normalised_water_df = normalised_water_df.drop(columns=features_to_drop)

# Save the new dataframe to a CSV file
selected_normalised_water_df.to_csv("selected_normalised_features_water_potability.csv", index=False)

print("==============================================================================================")
print("NORMALIZED DATA WITH SELECTED FEATURES")
print(selected_normalised_water_df)


# ============================================================================================================================================================================================
# MODEL DEVELOPMENT - DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define a function to train and evaluate a Decision Tree classifier
def train_and_evaluate_decision_tree(dataset_path, feature_cols, target_col='Potability'):
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Define features (X) and target (y)
    X = data[feature_cols]
    y = data[target_col]
    
    # Split dataset into training set and test set (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(random_state=1)
    
    # Train Decision Tree classifier
    clf = clf.fit(X_train, y_train)
    
    # Predict the response for the test dataset
    y_pred = clf.predict(X_test)
    
    # Calculate and return the accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

# Paths to your datasets
datasets = [
    "water_potability_no_outliers.csv",
    "normalised_water_potability.csv",
    "normalised_water_potability_with_composites.csv",
    "selected_normalised_features_water_potability.csv",
    "selected_features_water_potability.csv"
]

# Define the feature columns for each dataset
# Since 'Potability' is the target, it should be excluded from feature columns
all_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                'Organic_carbon', 'Trihalomethanes', 'Turbidity']

composite_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                      'Organic_carbon', 'Trihalomethanes', 'Turbidity',
                      'pH_Hardness_Cov', 'Sulfate_Conductivity_Cov',
                      'Turbidity_Organic_Carbon_Cov', 'Chloramines_Trihalomethanes_Cov']

selected_features = ['Solids', 'Chloramines', 'Conductivity', 'Organic_carbon', 'Trihalomethanes']

# Feature sets for each dataset
feature_sets = {
    "water_potability_no_outliers.csv": all_features,
    "normalised_water_potability.csv": all_features,
    "normalised_water_potability_with_composites.csv": composite_features,
    "selected_normalised_features_water_potability.csv": selected_features,
    "selected_features_water_potability.csv": selected_features
}

# Store accuracies for plotting
accuracies = []

# Train and evaluate the Decision Tree on each dataset, and store the accuracy
for dataset in datasets:
    accuracy = train_and_evaluate_decision_tree(dataset, feature_sets[dataset])
    accuracies.append(accuracy)
    print(f"Accuracy for {dataset}: {accuracy:.4f}")

# Plot the accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(datasets, accuracies, color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Decision Tree Classifier Accuracy for Different Datasets')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1

# Annotate bars with accuracy values
for bar, accuracy in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{accuracy:.4f}', ha='center', va='bottom')

plt.show()