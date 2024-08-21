import pandas as pd

# ACTIVITY 1 - DATA PREPARATION

# Paths to your CSV files
files = ['ampc/w1.csv', 'ampc/w2.csv', 'ampc/w3.csv', 'ampc/w4.csv']

# Read and combine all CSV files
dataframes = [pd.read_csv(file) for file in files]
combined_data = pd.concat(dataframes, ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('ampc/combined_data.csv', index=False)

# Shuffle the combined data
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save the shuffled data to another CSV file
shuffled_data.to_csv('ampc/all_data.csv', index=False)




# ACTIVITY 2 - MODEL TRAINING
print("==================================================================================================")
print("ACTIVITY 2")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('ampc/all_data.csv')

# Separate features (X) and target class (y)
# Assuming the last column is the target class
X = data.iloc[:, :-1]  # all rows, all columns except the last one
y = data.iloc[:, -1]   # all rows, last column only

# Splitting the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Create the SVM model
svm_model = SVC()

# Train the SVM model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy of the model on the test set
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(svm_model, X, y, cv=10)  # cv=10 for 10 folds
cross_val_accuracy = cv_scores.mean()  # Average of all cross-validation scores

# Output the results
print(f"Accuracy on Test Set: {test_accuracy}")
print(f"Cross-validation Scores: {cv_scores}")


# ACTIVITY 3 - HYPERPARAMETER TUNING

from sklearn.model_selection import GridSearchCV

print("==================================================================================================")
print("ACTIVITY 3")

# Set up the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Commonly used values for C
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100]  # Common values for gamma
}

# Create the SVM model with RBF kernel
svm_rbf = SVC(kernel='rbf')

# Create GridSearchCV to find the best parameters
grid_search = GridSearchCV(svm_rbf, param_grid, cv=10, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Extract the best parameters and model
best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 10-fold cross-validation with the best model
cv_scores = cross_val_score(best_svm_model, X, y, cv=10)

# Output the results
print(f"Best Parameters: {best_params}")
print(f"Best SVM model: {best_svm_model}")
print(f"Accuracy on Test Set: {test_accuracy}")
print(f"Cross-validation Accuracy: {cv_scores}")


# ACTIVITY 4 - FEATURE SELECTION

from sklearn.feature_selection import SelectKBest, f_classif

print("==================================================================================================")
print("ACTIVITY 4")

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=100)  # Select 100 best features based on ANOVA F-value
X_new = selector.fit_transform(X, y)

# Splitting the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.30, random_state=42)

# Create the SVM model with RBF kernel and best parameters
svm_rbf_tuned = SVC(kernel='rbf', **best_params)

# Train the SVM model
svm_rbf_tuned.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = svm_rbf_tuned.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 10-fold cross-validation with the tuned model
cv_scores = cross_val_score(svm_rbf_tuned, X_new, y, cv=10)
cross_val_accuracy = cv_scores.mean()

# Output the results
print(f"Accuracy on Test Set with 100 Best Features: {test_accuracy}")
print(f"Cross-validation Accuracy with 100 Best Features: {cross_val_accuracy}")




# ACTIVITY 5 - DIMENSIONALITY REDUCTION
print("==================================================================================================")
print("ACTIVITY 5")

from sklearn.decomposition import PCA

# Create the SVM model with RBF kernel and best parameters
svm_rbf_tuned = SVC(kernel='rbf', **best_params)

# Train the SVM model
svm_rbf_tuned.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = svm_rbf_tuned.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 10-fold cross-validation with the tuned model
cv_scores = cross_val_score(svm_rbf_tuned, X_pca, y, cv=10)
cross_val_accuracy = cv_scores.mean()

# Output the results
print(f"Accuracy on Test Set with PCA Features: {test_accuracy}")
print(f"Cross-validation Accuracy with PCA Features: {cross_val_accuracy}")