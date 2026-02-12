# Trained Models — Rainfall Pattern Analysis

> **Leakage-Free Models**: All models trained with strict temporal split and `.shift(1)` feature engineering to prevent data leakage.

## Model Files

| File | Model | Framework | Description |
|------|-------|-----------|-------------|
| `xgboost_model.pkl` | XGBoost | XGBoost | Gradient boosting with early stopping |
| `random_forest_model.pkl` | Random Forest | scikit-learn | Ensemble of decision trees (max_depth 15–20) |
| `lstm_model.h5` | LSTM | TensorFlow/Keras | Long Short-Term Memory recurrent network |
| `lightgbm_model.pkl` | LightGBM | LightGBM | Light gradient boosting machine |
| `gru_model.h5` | GRU | TensorFlow/Keras | Gated Recurrent Unit recurrent network |

## Support Files

- **scaler.pkl** — StandardScaler fitted on training data only (essential for inference)

## Training Configuration

- **Dataset**: NASA POWER API, 210 stations across India (8°N–34°N, 68°E–96°E)
- **Split**: Temporal — Train (2010–2019) / Validation (2020–2021) / Test (2022–2025)
- **Leakage Prevention**: All rainfall-derived features use `.shift(1)`, monthly climatology from training data only
- **Target**: Daily precipitation (mm)

## Usage

```python
import joblib
import tensorflow as tf

# Load tree-based model
xgb_model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Preprocess and predict
X_scaled = scaler.transform(features)
predictions = xgb_model.predict(X_scaled)

# Load deep learning model
lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
```

## Notes

- Models trained on Google Colab (GPU runtime)
- XGBoost and LightGBM are lightweight; LSTM and GRU require TensorFlow
- For detailed metrics, see `results/model_comparison.csv`
