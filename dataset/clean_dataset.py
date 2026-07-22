import pandas as pd

df = pd.read_csv('final_maternal_child_dataset.csv')

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


# 2) Explicit junk columns to remove (handle both plain and merged _x variants)
explicit_drop = {
    'v000', 'v001', 'v002', 'v003', 'v004', 'v005', 'v006', 'v007', 'v008', 'v008a',
    'v015', 'v016', 'v017', 'v018', 'v019', 'v021', 'v022', 'v023', 'v027', 'v028',
    'v030', 'v031', 'v032', 'v045a', 'v045b', 'v045c', 'v046', 'caseid', 'pidx', 'pidxb',
    'pord', 'midx', 'midxp', 'bidx', 'bidxp', 'bord', 'hidx', 'hidxa', 'hwidx', 'idx92',
    'idx92p', 'idx94', 'idx94p', 'idx96', 'is_in_children_recode', 'is_in_postnatal',
    'awfactt', 'awfactu', 'awfactr', 'awfacte', 'awfactw', 's228bd', 's228bm', 's228by'
}

# Convert plain names to actual present names in the dataframe
cols_to_drop = []
for col in df_cleaning.columns:
    col_base = col.replace('_x', '').replace('_y', '')
    if col in explicit_drop or col_base in explicit_drop:
        cols_to_drop.append(col)

# 3) Features with 0 sample (all values missing)
zero_sample = df_cleaning.columns[df_cleaning.isna().all()].tolist()

# 4) Missing percentage summary
missing_pct = df_cleaning.isna().mean().sort_values(ascending=False)

# 5) Features with >= 70% missing are removed directly
high_missing = missing_pct[missing_pct >= 0.70].index.tolist()
cols_to_drop.extend(zero_sample)
cols_to_drop.extend(high_missing)
cols_to_drop = list(dict.fromkeys(cols_to_drop))

# 6) Show review-only groups without removing them yet
missing_50_70 = missing_pct[(missing_pct >= 0.50) & (missing_pct < 0.70)]
missing_30_50 = missing_pct[(missing_pct >= 0.30) & (missing_pct < 0.50)]

print('Columns with 0 sample:', zero_sample)
print('Columns dropped because they have >= 70% missing:', high_missing)
print('Columns with 50-70% missing (review only):')
print(list(missing_50_70.index))
print('Columns with 30-50% missing (review only):')
print(list(missing_30_50.index))

# 7) Final cleaned dataframe and save to a new CSV on disk
clean_df = df_cleaning.drop(columns=cols_to_drop, errors='ignore')
clean_df.to_csv(output_csv, index=False)

print(f'Cleaned CSV saved to: {output_csv}')
print('Cleaned shape:', clean_df.shape)
clean_df.head()
