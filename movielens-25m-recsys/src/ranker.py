import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


FEATURES = ["mf_score", "item_count", "item_mean", "item_recency01", "genre_sim"] + list(genres_cols)

def build_table(parts, target_users, labels_df, max_rows=None):
    out, n = [], 0
    for p in parts:    # for each user (topk recommendation data)
        df = pd.read_parquet(p, columns=["u", "i", "mf_score"])    # data saved after scoring (SVD)
        df = df[df["u"].isin(target_users)]    # only target users, here, users with (ratings >= 4.0) data
        if df.empty:
            continue

        df = df.merge(item_features, on="i", how="left")    # merge recommendation list & item features on item
        ug = user_genre.reindex(df["u"].values).reset_index(drop=True)   # get user genre preference
        ug.columns = [c + "_user" for c in ug.columns]
        df = pd.concat([df.reset_index(drop=True), ug], axis=1)   # score data + item features + genre preference

        # here, ug columns are always the same for one user
        df["genre_sim"] = (df[genres_cols].values * df[[c + "_user" for c in genres_cols]].values).sum(axis=1).astype("float32")
        # print(df.head())
        df = df.merge(labels_df, on=["u", "i"], how="left")    # apply to valid/test set
        df["label"] = df["label"].fillna(0).astype("int8")

        usecol = ["u", "i", "label"] + FEATURES
        df = df[usecol].copy()
        df["mf_score"] = df["mf_score"].astype("float32")

        out.append(df)
        n += len(df)
        del df, ug
        gc.collect()

        if max_rows and n >= max_rows:
            break

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=use)



FEATURES_PLUS = ["mf_score","item_count","item_mean","item_recency01","genre_sim"] + list(genres_cols)

msk = (train_rank["u"] % 2 == 0)    # Even: True -> Train / Odd: False -> Valid
X_tr = train_rank.loc[msk, FEATURES_PLUS].values
y_tr = train_rank.loc[msk, "label"].values.astype("int8")
X_va = train_rank.loc[~msk, FEATURES_PLUS].values
y_va = train_rank.loc[~msk, "label"].values.astype("int8")

params = {
    "objective":"binary",
    "metrics":"auc",
    "learning_rate":0.05,
    "num_leaves":63,
    "min_date_in_leaf":50,
    "feature_fraction":0.8,
    "bagging_fraction":0.8,
    "bagging_freq":1,
    "verbosity":-1,
    "force_col_wise":True
}

dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)

# Train and Valid
gbm = lgb.train(
    params,
    dtrain,
    num_boost_round=1200,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)]
)

# Prediction
test_rank = test_rank.assign(
    pred = gbm.predict(test_rank[FEATURES_PLUS].values, num_iteration=gbm.best_iteration)
)