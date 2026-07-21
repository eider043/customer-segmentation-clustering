import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"Features escaladas: {X_scaled.shape}")
    return X_scaled, scaler