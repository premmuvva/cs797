#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing Data

# In[2]:


import pandas as pd

# Specify the path to your CSV file
excel_file_path = "C:\\Users\\singa\\Downloads\\UCS-Satellite-Database-Officialname-1-1-2023.xlsx"

# Use pandas to read the CSV file into a DataFrame
df = pd.read_excel(excel_file_path)

# Display the first few rows of the DataFrame
df.head()


# In[3]:


df.shape


# # Exploratory Data Analysis

# In[4]:


df.info()


# In[5]:


# Assuming 'df' is your DataFrame
columns_to_drop = df.columns[36:67]  # Column indices from 36 to 66
df.drop(columns=columns_to_drop, inplace=True)


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


columns_to_drop = df.columns[29:35]  # Column indices from 36 to 66
df.drop(columns=columns_to_drop, inplace=True)


# In[9]:


df.shape


# In[10]:


# Checking for unique values for the columns
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: \033[1m{unique_count} \033[0m")


# In[11]:


# Missing values for each columns

missing_values_count_per_column = {}

for col in df.columns:
    missing_values_count = df[col].isna().sum()
    missing_values_count_per_column[col] = missing_values_count

# Display the number of missing values for each column
for col, count in missing_values_count_per_column.items():
    print(f"{col}: \033[1m{count} \033[0m")


# In[12]:


duplicates = df.duplicated()

# Counting the number of duplicate rows
num_duplicates_rows = duplicates.sum()
print("Number of duplicate rows:", num_duplicates_rows)


# In[13]:


import pandas as pd

# Assuming 'df' is your DataFrame
threshold = 1000  # Set your threshold for missing values

# Calculate the number of missing values in each column
missing_values = df.isnull().sum()

# Filter columns with more than the threshold number of missing values
columns_to_drop = missing_values[missing_values > threshold].index

# Drop the selected columns
df.drop(columns=columns_to_drop, inplace=True)


# In[14]:


df.info()


# In[15]:


import pandas as pd

# Assuming 'df' is your DataFrame

# Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

# Sort the columns within each group
numerical_columns.sort()
categorical_columns.sort()

# Print the sorted columns
print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# In[16]:




# Calculating the number of rows and columns for subplots
num_rows = (len(numerical_columns) + 1) // 2
num_cols = 2

# Creating subplots with a shared y-axis
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))
fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

# Iterate through the numerical columns and creating histograms
for i, col_name in enumerate(numerical_columns):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]

    df[col_name].plot(kind='hist', bins=10, ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel(col_name)
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram for {col_name}')

    ax.tick_params(axis='x', rotation=45)

# Removing any empty subplots if there are an odd number of columns
if len(numerical_columns) % 2 == 1:
    fig.delaxes(axes[-1, -1])


# Adjusting the layout
plt.tight_layout()
# Showing the subplots
plt.show()


# In[17]:


# Creating Scatter plots for EDA

numerical_df = df[numerical_columns]
# Create cleaner scatter plots using Seaborn
sns.set(style="ticks")
sns.pairplot(numerical_df, height=2, aspect=1.5, diag_kind="kde")

# Add titles to the scatter plots
plt.suptitle('Pairwise Scatter Plots of Numerical Columns', fontsize=16)

# Show the scatter plots
plt.show()


# In[18]:


# finding the correlation between the column labels provided

# Calculating the correlation matrix
correlation_matrix = df.corr()

# Set the Seaborn style
sns.set(style="whitegrid")

# Create a correlation heatmap
plt.figure(figsize=(24, 16))  # Adjust the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# # Data Cleaning 

# ## Dropping columns with more than 1000 missing values

# In[19]:


missing_values = df.isna()

# Step 3: Sum missing values for each column
missing_counts = missing_values.sum()

# Step 4: Create a boolean mask for columns with more than 1000 missing values
columns_to_drop = missing_counts[missing_counts > 1000].index

# Step 5: Drop the columns
df = df.drop(columns=columns_to_drop)


# In[20]:


df.shape


# In[21]:


# Missing values for each columns

missing_values_count_per_column = {}

for col in df.columns:
    missing_values_count = df[col].isna().sum()
    missing_values_count_per_column[col] = missing_values_count

# Display the number of missing values for each column
for col, count in missing_values_count_per_column.items():
    print(f"{col}: \033[1m{count} \033[0m")


# In[22]:


df.info()


# In[23]:


import pandas as pd

# Assuming df is your DataFrame

# Select numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64'])

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object'])

print("Numerical Columns:")
print(numerical_columns.columns)

print("\nCategorical Columns:")
print(categorical_columns.columns)


# In[24]:


# Filling the missing values with the median values 
numerical_columns = df.select_dtypes(include=['float64', 'int64'])

# Fill missing values in numerical columns with the median
numerical_columns = numerical_columns.fillna(numerical_columns.median())

# Now, if you want to update the original DataFrame with the filled values:
df.update(numerical_columns)

# If you want to create a new DataFrame with the missing values filled:
# df_filled = df.copy()
# df_filled.update(numerical_columns)

# To confirm that missing values have been filled in the numerical columns:
print(df.head())


# In[25]:


df.head()


# In[26]:


# Remove rows with missing values in the "Type of Orbit" column
df = df.dropna(subset=['Type of Orbit'])

# This will remove all rows where "Type of Orbit" is missing.

# To reset the index after removing rows:
df = df.reset_index(drop=True)


# In[27]:


df.shape


# In[28]:


df['Class of Orbit'].unique()


# In[29]:


# Checking for unique values for the columns
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: \033[1m{unique_count} \033[0m")


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encode the 'Class of Orbit' column
df['Class of Orbit'] = le.fit_transform(df['Class of Orbit'])


# In[31]:


df['Class of Orbit'].value_counts()


# In[32]:


df['Class of Orbit'].unique()


# In[33]:


# Balacing the label as there is a Imbalanced data set for each categories

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Assuming df is your DataFrame

# Step 1: Select features and target column
features = df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]
target = df['Class of Orbit']

# Step 2: Define oversampling and undersampling strategies
oversample = RandomOverSampler(sampling_strategy='minority')  # Oversample the minority class
undersample = RandomUnderSampler(sampling_strategy='majority')  # Undersample the majority class

# Step 3: Create a pipeline for balancing
pipeline = Pipeline([
    ('oversample', oversample),
    ('undersample', undersample)
])

# Step 4: Apply the pipeline to balance the data
X_resampled, y_resampled = pipeline.fit_resample(features, target)

# Step 5: Create a new DataFrame with the balanced data
balanced_df = pd.DataFrame(X_resampled, columns=features.columns)
balanced_df['Class of Orbit'] = y_resampled

# Display the counts of each category in the balanced data
balanced_orbit_counts = balanced_df['Class of Orbit'].value_counts()
print(balanced_orbit_counts)


# In[34]:


pip install imbalanced-learn


# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assuming df is your DataFrame

# Step 1: Select features and target column
features = df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]
target = df['Class of Orbit']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE-based Over-sampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Create a new DataFrame with the balanced target values
resampled_df = pd.DataFrame(data=X_train_resampled, columns=features.columns)
resampled_df['Class of Orbit'] = y_train_resampled

# Step 5: Verify if the data is balanced
class_counts = resampled_df['Class of Orbit'].value_counts()
print(class_counts)


# In[36]:


resampled_df['Class of Orbit'].value_counts()


# In[37]:


resampled_df.shape


# In[38]:


resampled_df.head()


# # Random Forest Model

# In[39]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming df is your DataFrame

# Step 1: Select features and target column
features = resampled_df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]
target = resampled_df['Class of Orbit']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Choose a classification model (Random Forest in this example)
model = RandomForestClassifier(random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate testing accuracy
testing_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {training_accuracy}')
print(f'Testing Accuracy: {testing_accuracy}')


# In[40]:


# Select a random sample from the testing set
import random
sample_index = random.randint(0, len(X_test) - 1)
sample = X_test.iloc[[sample_index]]  # Create a DataFrame with a single sample

# Get the actual class of orbit for the selected sample
actual_class = y_test.iloc[sample_index]

# Make a prediction using the trained model
predicted_class = model.predict(sample)

# Display the results
print("Sample Features:")
print(sample)
print("Actual Class of Orbit:", actual_class)
print("Predicted Class of Orbit:", predicted_class[0])


# In[41]:


resampled_df['Class of Orbit'].value_counts()


# # Logistic Regression Model

# In[42]:



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming df is your preprocessed DataFrame

# Step 1: Select features and target column
features = resampled_df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]
target = resampled_df['Class of Orbit']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Calculate training accuracy
y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Step 5: Calculate testing accuracy
y_test_pred = model.predict(X_test)
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Step 6: Print training and testing accuracy
print(f'Training Accuracy: {training_accuracy}')
print(f'Testing Accuracy: {testing_accuracy}')


# In[43]:



# Step 1: Select a random sample from the testing set
sample_index = random.randint(0, len(X_test) - 1)
sample_features = X_test.iloc[sample_index]
actual_class = y_test.iloc[sample_index]

# Step 2: Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 3: Make a prediction using the model
predicted_class = model.predict([sample_features])

# Step 4: Display the results
print("Sample Features:")
print(sample_features)
print("Actual Class of Orbit:", actual_class)
print("Predicted Class of Orbit:", predicted_class[0])


# # Decision Tree Model

# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming df is your preprocessed DataFrame

# Step 1: Select features and target column
features = resampled_df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]
target = resampled_df['Class of Orbit']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Calculate training accuracy
y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Step 5: Calculate testing accuracy
y_test_pred = model.predict(X_test)
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Step 6: Print training and testing accuracy
print(f'Training Accuracy: {training_accuracy}')
print(f'Testing Accuracy: {testing_accuracy}')


# In[45]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random

# Assuming df is your preprocessed DataFrame

# Step 1: Select a random sample from the testing set
sample_index = random.randint(0, len(X_test) - 1)
sample = X_test.iloc[[sample_index]]  # Create a DataFrame with a single sample

# Step 2: Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 3: Make a prediction using the model
predicted_class = model.predict(sample)

# Step 4: Display the results
print("Sample Features:")
print(sample)
print("Actual Class of Orbit:", y_test.iloc[sample_index])
print("Predicted Class of Orbit:", predicted_class[0])


# ## Gray Box Attack

# ### To simulate a gray box attack, you can perturb the test data with noise

# In[46]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming 'df' contains your dataset with features and labels

# Separate the features (X) and labels (y)
X = resampled_df[['Longitude of GEO (degrees)', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Period (minutes)', 'Launch Mass (kg.)', 'NORAD Number']]  # Replace 'target_column' with the actual column name for your labels
y = resampled_df['Class of Orbit']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train your Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Evaluate the accuracy of the model on clean examples
y_pred = rf_model.predict(x_test)
clean_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Clean Examples: {clean_accuracy * 100:.2f}%")

# Implement feature engineering or preprocessing techniques to enhance robustness
# For example, you can apply feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Re-train the model with scaled features
rf_model.fit(x_train_scaled, y_train)

# Evaluate the accuracy of the model on clean examples after preprocessing
y_pred_scaled = rf_model.predict(x_test_scaled)
scaled_accuracy = accuracy_score(y_test, y_pred_scaled)
print(f"Accuracy on Clean Examples (with preprocessing): {scaled_accuracy * 100:.2f}%")

# To simulate a gray box attack, you can perturb the test data with noise
# You can add random noise to the features
epsilon = 0.8  # Adjust the magnitude of noise
x_test_perturbed = x_test_scaled + np.random.uniform(-epsilon, epsilon, size=x_test_scaled.shape)

# Evaluate the accuracy on perturbed examples
y_pred_perturbed = rf_model.predict(x_test_perturbed)
perturbed_accuracy = accuracy_score(y_test, y_pred_perturbed)
print(f"Accuracy on Perturbed Examples: {perturbed_accuracy * 100:.2f}%")


# In[47]:


import random

# Select a random data point
random_index = random.randint(0, len(x_test) - 1)
sample_clean = x_test.iloc[random_index]
sample_perturbed = x_test_perturbed[random_index]

# Display the random data points
print("Sample Clean Data:")
print(sample_clean)

print("\nSample Perturbed Data:")
print(sample_perturbed)

# Predict on the clean data
prediction_clean = rf_model.predict([sample_clean])[0]
accuracy_clean = "Correct" if prediction_clean == y_test.iloc[random_index] else "Incorrect"
print("\nPrediction on Clean Data:")
print(f"Predicted Class: {prediction_clean}")
print(f"Actual Class: {y_test.iloc[random_index]}")
print(f"Accuracy: {accuracy_clean}")

# Predict on the perturbed data
prediction_perturbed = rf_model.predict([sample_perturbed])[0]
accuracy_perturbed = "Correct" if prediction_perturbed == y_test.iloc[random_index] else "Incorrect"
print("\nPrediction on Perturbed Data:")
print(f"Predicted Class: {prediction_perturbed}")
print(f"Actual Class: {y_test.iloc[random_index]}")
print(f"Accuracy: {accuracy_perturbed}")


# In[48]:


pip install joblib


# In[49]:


from sklearn.ensemble import RandomForestClassifier
import joblib


# Save the model to a file
joblib.dump(rf_model, 'random_forest_model.joblib')


# In[50]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with the same random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with the same random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find the most prominent feature
most_prominent_feature_index = np.argmax(model.feature_importances_)

# Negate the values of the most prominent feature in the testing samples
X_test_adversarial = X_test.copy()
X_test_adversarial.iloc[:, most_prominent_feature_index] = -X_test_adversarial.iloc[:, most_prominent_feature_index]

# Predict using the model
y_pred_adversarial = model.predict(X_test_adversarial)

# Calculate the accuracy
adversarial_accuracy = accuracy_score(y_test, y_pred_adversarial)
print(f"Accuracy on Adversarial Examples: {adversarial_accuracy * 100:.2f}%")


# In[58]:


import numpy as np
from sklearn.metrics import accuracy_score

# Assuming 'X_test' contains your testing samples

# Manipulate a feature in the testing dataset
feature_index_to_manipulate = 3  # Change this to the index of the feature you want to manipulate

# Create a copy of the original testing dataset
X_test_manipulated = X_test.copy()

# Check if the number of samples matches
if X_test.shape[0] == X_test_manipulated.shape[0]:
    # Manipulate the specific feature with an arbitrary extreme value
    X_test_manipulated.iloc[:, feature_index_to_manipulate] = 800

    # Predict using the model
    y_pred_manipulated = model.predict(X_test_manipulated)

    # Calculate the accuracy
    manipulated_accuracy = accuracy_score(y_test, y_pred_manipulated)
    print(f"Accuracy on Manipulated Examples: {manipulated_accuracy * 100:.2f}%")
else:
    print("The number of samples in the manipulated dataset does not match the original dataset.")


# In[55]:


import numpy as np
from sklearn.metrics import accuracy_score

# Assuming 'X_test' contains your testing samples

# Define the feature you want to manipulate
feature_index_to_manipulate = 3  # Change this to the index of the feature you want to manipulate

# Define a range of threshold values to search
threshold_values = np.linspace(-1000, 1000, 10000)  # Adjust the range and number of values as needed

# Initialize variables to track the best threshold and accuracy
best_threshold = None
best_accuracy = 1.0  # Initialize with a high accuracy value
target_accuracy = 0.20  # Target accuracy (20%)

# Iterate through the threshold values
for threshold in threshold_values:
    X_test_manipulated = X_test.copy()
    X_test_manipulated.iloc[:, feature_index_to_manipulate] = threshold

    # Predict using the model
    y_pred_manipulated = model.predict(X_test_manipulated)

    # Calculate the accuracy
    manipulated_accuracy = accuracy_score(y_test, y_pred_manipulated)

    # Check if the accuracy is close to the target
    if abs(manipulated_accuracy - target_accuracy) < abs(best_accuracy - target_accuracy):
        best_accuracy = manipulated_accuracy
        best_threshold = threshold

    # Break early if the accuracy is below the target
    if manipulated_accuracy <= target_accuracy:
        break

# Print the best threshold and achieved accuracy
print(f"Best Threshold: {best_threshold}")
print(f"Accuracy at Best Threshold: {best_accuracy * 100:.2f}%")


# In[64]:


df['Eccentricity'].describe()


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Assuming 'X_test' contains your testing samples

# Define the feature you want to manipulate
feature_index_to_manipulate = 3  # Change this to the index of the feature you want to manipulate

# Define a range of threshold values to search
threshold_values = np.linspace(-1000, 1000, 10000)  # Adjust the range and number of values as needed

# Initialize lists to store accuracy values
accuracies_original = []
accuracies_manipulated = []

# Iterate through the threshold values
for threshold in threshold_values:
    X_test_manipulated = X_test.copy()
    X_test_manipulated.iloc[:, feature_index_to_manipulate] = threshold

    # Predict using the model
    y_pred_original = model.predict(X_test)
    y_pred_manipulated = model.predict(X_test_manipulated)

    # Calculate the accuracy
    accuracy_original = accuracy_score(y_test, y_pred_original)
    accuracy_manipulated = accuracy_score(y_test, y_pred_manipulated)

    accuracies_original.append(accuracy_original)
    accuracies_manipulated.append(accuracy_manipulated)

# Plot the accuracy before and after the attack
plt.figure(figsize=(10, 6))
plt.plot(threshold_values, accuracies_original, label='Original Accuracy')
plt.plot(threshold_values, accuracies_manipulated, label='Manipulated Accuracy')
plt.xlabel('Threshold Value')
plt.ylabel('Accuracy')
plt.title('Effect of Attack on Model Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




