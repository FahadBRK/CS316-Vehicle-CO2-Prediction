"""Interpretable fuel features, learned exclusively from training records."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NUMERIC = ["Engine size (L)", "Cylinders", "Combined (L/100 km)"]
FUEL = "Fuel type"


class FuelInteractionFeatures(TransformerMixin, BaseEstimator):
    """Allow each observed fuel to have its own intercept and fuel-use slope.

    X (regular gasoline) is the reference when present. Unknown fuel codes
    receive the reference encoding during validation, so evaluation remains
    defined even when a rare fuel is absent from one training fold. The CLI
    rejects such codes for new user-supplied predictions. No target is used.
    """

    def fit(self, X, y=None):
        self.medians_ = X[NUMERIC].astype(float).median().to_numpy()
        self.fuels_ = sorted(X[FUEL].astype(str).unique())
        self.reference_ = "X" if "X" in self.fuels_ else self.fuels_[0]
        self.other_fuels_ = [f for f in self.fuels_ if f != self.reference_]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        numeric = X[NUMERIC].to_numpy(dtype=np.float64, copy=True)
        # Vectorized filling avoids Python loops over individual vehicle rows.
        missing = np.isnan(numeric)
        numeric[missing] = np.broadcast_to(self.medians_, numeric.shape)[missing]
        fuel = X[FUEL].astype(str).to_numpy()
        flags = (np.column_stack([fuel == f for f in self.other_fuels_]).astype(float)
                 if self.other_fuels_ else np.empty((len(numeric), 0)))
        interactions = numeric[:, 2:3] * flags
        return np.column_stack([numeric, flags, interactions])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "medians_")
        return np.array(NUMERIC + [f"fuel_{f}" for f in self.other_fuels_]
                        + [f"combined_x_{f}" for f in self.other_fuels_], dtype=object)
