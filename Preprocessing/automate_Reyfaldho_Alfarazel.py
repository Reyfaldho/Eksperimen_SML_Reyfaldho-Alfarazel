import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "https://raw.githubusercontent.com/Reyfaldho/Eksperimen_SML_Reyfaldho-Alfarazel/refs/heads/main/Heart%20Disease_raw/heart_disease_uci.csv"
OUTPUT_DIR = "preprocessing"
OUTPUT_FILENAME = "dataset_HeartDisease_membangun_sistem_machine_learning_preprocessing.csv"

NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL_FEATURES = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
REQUIRED_COLS = ['num'] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def preprocess_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except Exception:
        return pd.DataFrame()

    if any(col not in df.columns for col in REQUIRED_COLS):
        return pd.DataFrame()

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df['num'] = pd.to_numeric(df['num'], errors='coerce')
    df = df.dropna(subset=['num']).reset_index(drop=True)

    df['target'] = (df['num'] > 0).astype(int)
    df['num_original'] = df['num']

    for col in NUMERICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUMERICAL_FEATURES])
    X_num_df = pd.DataFrame(X_num, columns=NUMERICAL_FEATURES)

    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    X_cat = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_names)

    processed_df = pd.concat([X_num_df, X_cat_df], axis=1)
    processed_df['target'] = df['target'].values
    processed_df['num_original'] = df['num_original'].values

    return processed_df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_df = preprocess_data(DATA_PATH)

    if not processed_df.empty:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        processed_df.to_csv(output_path, index=False)
