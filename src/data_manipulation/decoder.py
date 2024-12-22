import json
import pandas as pd

def denormalize_column(df, column, denormalization_config):
    """Sayısal bir sütunu orijinal ölçeğine geri dönüştürür."""
    min_value = denormalization_config[column]['min']
    max_value = denormalization_config[column]['max']
    df[column] = (df[column] * (max_value - min_value) + min_value).round(2)
    return df

def decode_column(df, column, denormalization_config):
    """Kategorik bir sütunu orijinal kategorik değerlerine geri döndürür."""
    if denormalization_config[column]['type'] == 'direct':
        # Mapping ile doğrudan dönüşüm
        mapping = {float(k): v for k, v in denormalization_config[column]['mapping'].items()}
        df[column] = df[column].map(mapping)
    elif denormalization_config[column]['type'] == 'one_hot':
        # One-hot encoding dönüşümü
        prefix = denormalization_config[column]['prefix']
        categories = denormalization_config[column]['categories']
        one_hot_columns = [f"{prefix}_{cat}" for cat in categories]
        print(f"one_hot_columns: {one_hot_columns} for column: {column}")
        
        # En yüksek değeri alıp kategoriye geri dönüştür
        df[column] = (
            df[one_hot_columns]
            .idxmax(axis=1)  # En yüksek değerli sütunu seç
            .str.replace(f"{prefix}_", "")  # Prefix'i kaldır
        )
        # One-hot sütunlarını temizle (isteğe bağlı)
        df.drop(columns=one_hot_columns, inplace=True)
    return df

def decode_dataframe(df, denormalization_config_path):
    """Veri setini verilen yapılandırmaya göre orijinal haline geri döndürür."""
    # JSON yapılandırmasını yükle
    with open(denormalization_config_path, 'r') as file:
        denormalization_config = json.load(file)
    
    for column in denormalization_config:
        if denormalization_config[column]['isNumerical']:
            df = denormalize_column(df, column, denormalization_config)
        else:
            df = decode_column(df, column, denormalization_config)
        
    return df

df = pd.read_csv("datasets/validation/validation_data_ID_numeric.csv")
decoded_df = decode_dataframe(df, "config_files/denormalization_config.json")
#move target feature to the end
target_feature = 'loan_status'
target_feature_index = df.columns.get_loc(target_feature)
decoded_df = decoded_df[[col for col in decoded_df.columns if col != target_feature] + [target_feature]]

decoded_df.to_csv("datasets/validation/validation_data_ID_decoded.csv", index=False)