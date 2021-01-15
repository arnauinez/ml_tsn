from sklearn import preprocessing
import pandas as pd

class Tools:

    @staticmethod
    def get_feasibility(df):
        feasibility = df["valid"].value_counts(normalize=True).mul(100)
        return feasibility

    @staticmethod
    def min_max_scaler(df):
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        df[6] = df[6].astype(int)
        return df

    @staticmethod
    def get_X_y(df):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y