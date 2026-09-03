import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove identifier/export columns because they do not describe customer behavior.
columns_to_drop = [column for column in ["Unnamed: 0", "CustomerID"] if column in df.columns]
df = df.drop(columns=columns_to_drop)

# Standardize inconsistent category labels found in the source data.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove exact duplicates and validate the cleaned data.
df = df.drop_duplicates().reset_index(drop=True)
if df.isnull().any().any():
    missing = df.isnull().sum()
    raise ValueError(f"Missing values remain after cleaning:\n{missing[missing > 0]}")

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Stratification keeps the imbalanced purchase ratio consistent across splits.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Training rows: {len(Xtrain)}, Testing rows: {len(Xtest)}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts(normalize=True).round(4))
