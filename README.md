# collab-sol

Modeling code for next-year grade-crossing risk and conditional severity.

The `main` branch owns temporal splitting, shared model interfaces, training,
Bayesian hyperparameter tuning, and evaluation. Reproducible source-data
construction is retained separately on the `dataset-pipeline` branch.

The planned model panel is logistic regression, XGBoost, LightGBM, an MLP, and
an FT-Transformer. Each model implements the `BaseModel` interface so the same
temporal cross-validation and Bayesian tuning pipeline can train and score it.
