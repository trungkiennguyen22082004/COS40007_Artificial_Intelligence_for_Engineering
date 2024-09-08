import pandas as pd
import matplotlib.pyplot as plt

# ACTIVITY 1: CLASS LABELLING/CREATING GROUND TRUTH DATA

# Step 1: Convert the numerical value strength to a categorial value
#   Step 1.1: Load the CSV dataset into a pandas DataFrame
concrete_data = pd.read_csv("concrete.csv")

#   Step 1.2: Define a function to categorize the strength values
def categorize_strength(strength):
    if strength < 20:
        return 1  # Very low
    elif 20 <= strength < 30:
        return 2  # Low
    elif 30 <= strength < 40:
        return 3  # Moderate
    elif 40 <= strength < 50:
        return 4  # Strong
    else:
        return 5  # Very strong
    
#   Step 1.3: Apply the function to the 'strength' column to create a new column
concrete_data["strength_category"] = concrete_data["strength"].apply(categorize_strength)

#   Step 1.4: Write the converted DataFrame to a new CSV file
concrete_data.to_csv("converted_concrete.csv", index=False)

# Step 2: Visualize the distribution of classes in a bar chart
class_distribution = concrete_data['strength_category'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Strength Categories')
plt.xlabel('Strength Category')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
# plt.show()

# There is no imbalance between any 2 classes in the dataset


# ACTIVITY 2: FEATURE ENGINEERING

# Step 1: Simplify the ‘age’ feature

#   Step 1.1: Find the unique age values and their counts
unique_ages = concrete_data["age"].value_counts().sort_index()

#   Step 1.2: Convert the 'age' feature to categorical values
age_mapping = {}
for i, age in enumerate(unique_ages.index):
    # Map each unique age to a categorical value (starting from 1)
    age_mapping[age] = i + 1

concrete_data["age_category"] = concrete_data["age"].map(age_mapping)

print("Unique Age Values and Counts:")
print(unique_ages)
print("------------------------------------------------------------------------------------------------------")
print("Age Mapping:")
print(age_mapping)
print("------------------------------------------------------------------------------------------------------")
print("Concreate data based on age:")
print(concrete_data[["age", "age_category"]].to_string())

