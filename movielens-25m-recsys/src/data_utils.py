import pandas as pd
from sklearn.preprocessing import LabelEncoder

le_u, le_i = LabelEncoder(), LabelEncoder()
train["u"] = le_u.fit_transform(train["userId"])
train["i"] = le_i.fit_transform(train["movieId"])

def kcore(df, k=10):
    d = df
    while True:
        before = len(d)
        d = d[d.groupby("userId")["userId"].transform("count") >=k ]    # users more than x10
        d = d[d.groupby("movieId")["movieId"].transform("count") >=k ]  # movies rated more than x10
        after = len(d)
        if before==after: 
            break
    return d.reset_index(drop=True)


def encode_u_i(df):
    d = df[df["userId"].isin(le_u.classes_) & df["movieId"].isin(le_i.classes_)].copy()
    
    d["u"] = le_u.transform(d["userId"])
    d["i"] = le_i.transform(d["movieId"])

    return d