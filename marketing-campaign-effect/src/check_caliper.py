# Check Caliper

# LogisticRegression

# X_df = df[covariates].copy()
# X = X_df.to_numpy()
# T = df['Response'].astype(int).to_numpy()

# lr = LogisticRegression(max_iter=3000)
# lr.fit(X, T)
# ps = lr.predict_proba(X)[:, 1]

# idx_t = np.where(T == 1)[0]
# idx_c = np.where(T == 0)[0]
# ps_t, ps_c = ps[idx_t], ps[idx_c]

# nn = NearestNeighbors(n_neighbors=1).fit(ps_c.reshape(-1, 1))
# dist, idx = nn.kneighbors(ps_t.reshape(-1, 1))
# dist = dist.ravel()

# print(f"(dist.max()): {dist.max()}")
# print(f"distance distribution (75%, 80%, 90%): {np.quantile(dist, [0.75, 0.8, 0.9])}")


# XGBoost

# X_df = df[covariates].copy()
# X = X_df.to_numpy()
# T = df['Response'].astype(int).to_numpy()

# lr = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
#                     use_label_encoder=False, eval_metric='logloss', random_state=42)
# lr.fit(X, T)
# ps = lr.predict_proba(X)[:, 1]

# idx_t = np.where(T == 1)[0]
# idx_c = np.where(T == 0)[0]
# ps_t, ps_c = ps[idx_t], ps[idx_c]

# nn = NearestNeighbors(n_neighbors=1).fit(ps_c.reshape(-1, 1))
# dist, idx = nn.kneighbors(ps_t.reshape(-1, 1))
# dist = dist.ravel()

# print(f"(dist.max()): {dist.max()}")
# print(f"distance distribution (75%, 80%, 90%): {np.quantile(dist, [0.75, 0.8, 0.9])}")