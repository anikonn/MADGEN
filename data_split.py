from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("./data/canopus/raw/canopus.csv")
# smiles = data['SMILES']
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_df["source"] = "train"
validation_df["source"] = "val"
test_df["source"] = "test"

combined_df = pd.concat([train_df, validation_df, test_df], axis=0)

combined_df = combined_df.sample(frac=1).reset_index(drop=True)
combined_df.to_csv("./data/canopus/raw/canopus.csv")
