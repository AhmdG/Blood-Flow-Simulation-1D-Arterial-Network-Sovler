import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assume X_raw is your input array of shape (n_samples, 13)
# Let's define the column indices for each group
continuous_bounded = [0, 1]        # Features 0, 1
categorical_indices = [2, 3, 4]    # Features 2, 3, 4
continuous_small = [5, 6, 7, 8, 9, 10]  # Features 5-10
continuous_large = [11, 12]        # Features 11, 12

# Create a column transformer to handle each group appropriately
preprocessor = ColumnTransformer(
    transformers=[
        ('bounded', MinMaxScaler(), continuous_bounded),
        ('categories', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_indices),
        ('small_scale', StandardScaler(), continuous_small),
        ('large_scale', StandardScaler(), continuous_large)
    ],
    remainder='passthrough'  # This shouldn't be needed as we've defined all columns
)

# Fit on TRAINING data only
preprocessor.fit(X_train)

# Transform all data
X_processed = preprocessor.transform(X_raw)

from sklearn.preprocessing import RobustScaler
import numpy as np

# Use robust scaling with pre-defined quantiles
robust_scaler = RobustScaler(quantile_range=(25, 75))

# But you still need to fit it on some data first!
# If you have some representative data, even a small sample:
representative_sample = np.array([...])  # Some sample data
robust_scaler.fit(representative_sample)