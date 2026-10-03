import pandas as pd
import numpy as np
import re
import joblib
from lightgbm import LGBMRegressor

# ---- Load & clean (same as your notebook) ----
df = pd.read_csv("zomato.csv", sep=',', engine='python', on_bad_lines='warn')
df = df.drop(['url', 'dish_liked', 'phone'], axis=1)
df.rename({
    'approx_cost(for two people)': 'cost_for_2',
    'listed_in(type)': 'listed_in_type',
    'listed_in(city)': 'listed_in_city'
}, axis=1, inplace=True)

df.votes = pd.to_numeric(df.votes, errors='coerce').fillna(0).astype('int')

df = df.loc[df.rate != 'NEW']
df = df.loc[df.rate != '-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if isinstance(x, str) else x
df.rate = df.rate.apply(remove_slash).str.strip().astype('float')

df['cost_for_2'] = df['cost_for_2'].astype(str).str.replace(',', '', regex=False)
df['cost_for_2'] = pd.to_numeric(df['cost_for_2'], errors='coerce')

df['rate'] = df['rate'].fillna(df['rate'].mean())
df['cost_for_2'] = df['cost_for_2'].fillna(df['cost_for_2'].median())
df['location'] = df['location'].fillna('Unknown')
df['rest_type'] = df['rest_type'].fillna('Unknown')
df['cuisines'] = df['cuisines'].fillna('Unknown')

# ---- Feature selection ----
x = df[['online_order', 'book_table', 'votes', 'location', 'rest_type',
        'cuisines', 'cost_for_2', 'listed_in_type', 'listed_in_city', 'name']]
y = df['rate']

low_card_cols = ['online_order', 'book_table', 'rest_type', 'listed_in_type']
high_card_cols = ['location', 'cuisines', 'listed_in_city', 'name']

# ---- Target encoding maps (SAVE these — Streamlit needs them at inference time) ----
target_encoding_maps = {}
df_model = x.copy()
df_model['rate'] = y

for col in high_card_cols:
    means = df_model.groupby(col)['rate'].mean()
    target_encoding_maps[col] = means.to_dict()
    df_model[col] = df_model[col].map(means)

global_mean = y.mean()

# ---- One-hot encode ----
df_model = pd.get_dummies(df_model, columns=low_card_cols, drop_first=True)

x_final = df_model.drop('rate', axis=1)
y_final = df_model['rate']

# ---- Clean column names (LightGBM requirement) ----
def clean_column_names(dframe):
    dframe.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in dframe.columns]
    return dframe

x_final = clean_column_names(x_final)
final_columns = x_final.columns.tolist()   # SAVE this — needed to align input at inference time

# ---- Train final model on FULL data (for deployment, use all data you have) ----
model = LGBMRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    num_leaves=31, random_state=42, n_jobs=-1
)
model.fit(x_final, y_final)

# ---- Save everything the Streamlit app needs ----
joblib.dump(model, 'model.pkl')
joblib.dump(target_encoding_maps, 'target_encoding_maps.pkl')
joblib.dump(global_mean, 'global_mean.pkl')
joblib.dump(final_columns, 'final_columns.pkl')

# also save the raw unique values for building dropdowns in the UI
dropdown_values = {
    'online_order': sorted(df['online_order'].unique().tolist()),
    'book_table': sorted(df['book_table'].unique().tolist()),
    'rest_type': sorted(df['rest_type'].unique().tolist()),
    'listed_in_type': sorted(df['listed_in_type'].unique().tolist()),
    'location': sorted(df['location'].unique().tolist()),
    'cuisines': sorted(df['cuisines'].unique().tolist())[:200],  # trim if huge
    'listed_in_city': sorted(df['listed_in_city'].unique().tolist()),
    'name': sorted(df['name'].unique().tolist())[:500],  # trim if huge
}
joblib.dump(dropdown_values, 'dropdown_values.pkl')

print("Saved: model.pkl, target_encoding_maps.pkl, global_mean.pkl, final_columns.pkl, dropdown_values.pkl")