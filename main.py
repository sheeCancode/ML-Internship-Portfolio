import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# 2. Basic EDA: Look at the first 5 rows
print("--- Dataset Head ---")
print(df.head())

# 3. Check for missing values (crucial for 'clean structure')
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Simple Visualization: Price Distribution
df['MedHouseVal'].hist(bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price (in $100k)')
plt.ylabel('Frequency')
plt.show()
input("\nExecution finished. Press Enter to close...")