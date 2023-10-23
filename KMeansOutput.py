#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[4]:


# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)


# In[ ]:





# In[49]:


# !python -m pip install openpyxl
get_ipython().system('pip install kneed')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
# !conda update -n base -c defaults conda -y
# !conda install openpyxl -y
# !pip install openpyxl  --upgrade
get_ipython().system('pip install pandas --upgrade')
get_ipython().system('pip install scikit-learn')


# In[7]:






# In[3]:


# Specify the path to your CSV file
excel_file_path = "UCS-Satellite-Database-1-1-2023.xlsx"
# "/Users/reddy/Downloads/UCS-Satellite-Database-1-1-2023.xlsx"

# Use pandas to read the CSV file into a DataFrame
df_org = pd.read_excel(excel_file_path, engine='openpyxl')

# Display the first few rows of the DataFrame
df_org.head()


# In[4]:


df_org.shape


# In[5]:


df_org.info()


# In[6]:


columns_to_drop = df_org.columns[27:]  # Column indices from 36 to 66
df = df_org.drop(columns=columns_to_drop)
df.info()


# In[7]:


for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: \033[1m{unique_count} \033[0m")


# In[8]:


missing_values_count_per_column = {}

for col in df.columns:
    missing_values_count = df[col].isna().sum()
    missing_values_count_per_column[col] = missing_values_count

# Display the number of missing values for each column
for col, count in missing_values_count_per_column.items():
    print(f"{col}: \033[1m{count} \033[0m")


# In[9]:


duplicates = df.duplicated()

# Counting the number of duplicate rows
num_duplicates_rows = duplicates.sum()
print("Number of duplicate rows:", num_duplicates_rows)



# In[10]:


import pandas as pd

# Assuming 'df' is your DataFrame
threshold = 1000  # Set your threshold for missing values

# Calculate the number of missing values in each column
missing_values = df.isnull().sum()

# Filter columns with more than the threshold number of missing values
columns_to_drop = missing_values[missing_values > threshold].index

# Drop the selected columns
df.drop(columns=columns_to_drop, inplace=True)


# In[11]:


df.info()


# In[12]:


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



# In[13]:


scaled_features = df.drop(categorical_columns, axis=1)
scaled_features.info()
final_scaled_features = scaled_features.dropna()


# In[ ]:





# In[14]:


from sklearn.cluster import KMeans
kmeans = KMeans(
     init="random",
     n_clusters=3,
     n_init=10,
     max_iter=300,
     random_state=42)
kmeans.fit(final_scaled_features)


# In[40]:


kmeans.inertia_


# In[41]:


kmeans.cluster_centers_


# In[42]:


kmeans.n_iter_


# In[43]:


kmeans.labels_[:5]


# In[59]:


kmeans_kwargs = {
     "init": "random",
     "n_init": 10,
     "max_iter": 300,
     "random_state": 42,
}


 # A list holds the SSE values for each k
sse = []
for k in range(1, 15):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(final_scaled_features)
     sse.append(kmeans.inertia_)


# In[60]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 15), sse)
plt.xticks(range(1, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[51]:


from kneed import KneeLocator
kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
kl.elbow


# In[57]:


from sklearn.metrics import silhouette_score

silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(final_scaled_features)
    score = silhouette_score(final_scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[58]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 15), silhouette_coefficients)
plt.xticks(range(2, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

