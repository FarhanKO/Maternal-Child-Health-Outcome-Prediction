import pandas as pd

df = pd.read_csv('final_maternal_child_dataset.csv')
df.head()

# Read the CSV and prepare a cleaned output file
input_csv = 'final_maternal_child_dataset.csv'
output_csv = 'final_maternal_child_dataset_cleaned.csv'

df = pd.read_csv(input_csv)
df_cleaning = df.copy()

# 1) Remove duplicate merge artifacts (_x/_y)
for x_col in [c for c in df_cleaning.columns if c.endswith('_x')]:
    base = x_col[:-2]
    y_col = f"{base}_y"
    if y_col in df_cleaning.columns:
        if df_cleaning[x_col].equals(df_cleaning[y_col]):
            df_cleaning = df_cleaning.drop(columns=[y_col], errors='ignore')
        else:
            df_cleaning[x_col] = df_cleaning[x_col].combine_first(df_cleaning[y_col])
            df_cleaning = df_cleaning.drop(columns=[y_col], errors='ignore')

