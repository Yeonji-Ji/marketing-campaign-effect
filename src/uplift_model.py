import numpy as np
from lightgbm import LGBMRegressor

class XLearner:
    """
    - fit: outcome model(μ0, μ1) → imputed effect(d0, d1) → effect model(τ0, τ1)
    - predict: return τ(x) 
    """
    def __init__(self, mu0=None, mu1=None, tau0=None, tau1=None, random_state: int = 42):
        self.mu0 = mu0 or LGBMRegressor(verbose=-1,n_estimators=400)  # type: ignore
        self.mu1 = mu1 or LGBMRegressor(verbose=-1,n_estimators=400)  # type: ignore
        self.tau0 = tau0 or LGBMRegressor(verbose=-1,n_estimators=400)  # type: ignore
        self.tau1 = tau1 or LGBMRegressor(verbose=-1,n_estimators=400)  # type: ignore
        self.p_t = 0.5
        self.random_state = random_state

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray):

        X0, y0 = X[T == 0], y[T == 0]
        X1, y1 = X[T == 1], y[T == 1]

        self.mu0.fit(X0, y0)    # Use only Control data
        self.mu1.fit(X1, y1)    # Use only Treatment data

        # Imputed effects: 
        d0 = self.mu1.predict(X0) - y0
        d1 = y1 - self.mu0.predict(X1)

        # 3) Effect models(τ0, τ1)
        self.tau0.fit(X0, d0)    # Use X0 and Predict d0
        self.tau1.fit(X1, d1)    # Use X1 and Predict d1

        self.p_t = float(T.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Final CATE prediction: (1-p_t)*τ0(x) + p_t*τ1(x)"""
        w0 = 1 - self.p_t
        w1 = self.p_t
        return w0 * self.tau0.predict(X) + w1 * self.tau1.predict(X)
    

    def fit_xlearner(X: np.ndarray, T: np.ndarray, y: np.ndarray) -> XLearner:
        return XLearner().fit(X, T, y)
    

    def predict_cate(model: XLearner, X_new: np.ndarray) -> np.ndarray:
        return model.predict(X_new)
    

 