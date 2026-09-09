   #Use the project folder name, then the folder where model building code will be stored.

import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # complete the code: path to the registered tourism.csv inside the data folder

#df.drop(columns=[_______], inplace=True)
# complete the code: drop the customer identifier column, it is not a predictive feature

columns_to_drop = [column for column in ["Unnamed: 0", "CustomerID"] if column in df.columns]
df = df.drop(columns=columns_to_drop)

# Standardize inconsistent category labels found in the source data.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove exact duplicates and validate the cleaned data.
df = df.drop_duplicates().reset_index(drop=True)
if df.isnull().any().any():
    missing = df.isnull().sum()
    raise ValueError(f"Missing values remain after cleaning:\n{missing[missing > 0]}")

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target =  "ProdTaken"  # complete the code to set the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # complete the code: which variable should stay balanced across the splits?
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


#print(ytrain.value_counts())

print("Data prepared: train/test splits written.")
print(f"Training rows: {len(Xtrain)}, Testing rows: {len(Xtest)}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts(normalize=True).round(4))
