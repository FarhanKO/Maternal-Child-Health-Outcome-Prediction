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


# 2) Explicit junk columns to remove (handle both plain and merged _x variants)
explicit_drop = {
    'v000', 'v001', 'v002', 'v003', 'v004', 'v005', 'v006', 'v007', 'v008', 'v008a',
    'v015', 'v016', 'v017', 'v018', 'v019', 'v021', 'v022', 'v023', 'v027', 'v028',
    'v030', 'v031', 'v032', 'v045a', 'v045b', 'v045c', 'v046', 'caseid', 'pidx', 'pidxb',
    'pord', 'midx', 'midxp', 'bidx', 'bidxp', 'bord', 'hidx', 'hidxa', 'hwidx', 'idx92',
    'idx92p', 'idx94', 'idx94p', 'idx96', 'is_in_children_recode', 'is_in_postnatal',
    'awfactt', 'awfactu', 'awfactr', 'awfacte', 'awfactw', 's228bd', 's228bm', 's228by'
}

