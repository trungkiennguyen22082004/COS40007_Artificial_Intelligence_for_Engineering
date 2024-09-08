import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np

# UNDERSTANDING DATA
#   Load the dataset
data = pd.read_csv('water_potability.csv')

print("==============================================================================================")
print("ORIGINAL DATA")
print(data)

#   Number of rows and columns
num_rows, num_columns = data.shape
# print(f"Number of rows: {num_rows}")
# print(f"Number of columns: {num_columns}")

#   Column names and data types
column_info = data.dtypes
# print(column_info)

# DATA CLEANING
#   Remove duplicate entries
data_cleaned = data.drop_duplicates()

#   Check for outliers using boxplots for each column
# plt.figure(figsize=(15, 10))
# data_cleaned.boxplot()
# plt.title('Boxplot to Detect Outliers')
# plt.xticks(rotation=45)
# plt.show()

#   Remove outliers (Assume outliers are values beyond 1.5*IQR from Q1 and Q3)
Q1 = data_cleaned.quantile(0.25)
Q3 = data_cleaned.quantile(0.75)
IQR = Q3 - Q1

#   Define outliers as values that are beyond the IQR range
outliers_condition = ~((data_cleaned < (Q1 - 1.5 * IQR)) | (data_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)
data_no_outliers = data_cleaned[outliers_condition]

#   Boxplot after removing outliers to verify correction
# plt.figure(figsize=(15, 10))
# data_no_outliers.boxplot()
# plt.title('Boxplot After Removing Outliers')
# plt.xticks(rotation=45)
# plt.show()

#   Check for any missing values
missing_values = data_no_outliers.isnull().sum()

print("==============================================================================================")
print("MISSING VALUES")
print(missing_values)

data_no_outliers.to_csv("water_potability_no_outliers.csv", index=False)
data = data_no_outliers
print("==============================================================================================")
print("DATA WITH NO OUTLIERS")
print(data)

# EXPLORATORY DATA ANALYSIS

#   Univariate Analysis - Visualize the distribution of each feature 

#       Define a function to plot the distribution with mean line
def plot_distribution(data, column_name):
    plt.figure(figsize=(13, 6))
    sns.histplot(data[column_name], color="b", kde=True)
    plt.axvline(data[column_name].mean(), linestyle="dashed", color="k", label="mean", linewidth=2)
    plt.legend(loc="best", prop={"size": 14})
    plt.title(f"{column_name} Distribution")
    plt.show()

#       List of columns to plot (excluding 'Potability')
columns = [col for col in data_no_outliers.columns if col != 'Potability']

#       Plot each column
# for col in columns:
#     plot_distribution(data_no_outliers, col)


#   Summary statistics
print("==============================================================================================")
print("SUMMARY STATISTICS")
print(data.describe().T)

#   Multivariate Analysis - Visualize the grid of pairwise plots
sns.pairplot(data_no_outliers, diag_kind='kde', corner=True)
# plt.show()

#   Multivariate Analysis - Correlations values
print("==============================================================================================")
print("CORRELATION VALUES")
print(data.corr())

#   Correlations Heatmap
#       Calculate the absolute correlation matrix
corr = abs(data.corr())

#       Mask the upper triangle of the heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))

#       Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', center=0)
plt.title('Heatmap of Pairwise Correlations Among Variables')

plt.show()
