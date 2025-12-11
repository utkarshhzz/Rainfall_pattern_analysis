# Trained Models

This folder contains the trained models achieving 99.98% R² accuracy.

## Model Files

| File | Model | Test R² | Test RMSE | Size |
|------|-------|---------|-----------|------|
| `widedeep_model.pt` | Wide & Deep Network (Best) | 99.98% | 0.41 mm | 354 KB |
| `lstm_model.pt` | LSTM | 99.76% | 1.40 mm | 3.9 MB |
| `deepnet_model.pt` | DeepNet | 99.81% | 1.25 mm | 2.1 MB |
| `xgboost_model.json` | XGBoost | 99.90% | 0.91 mm | 83 MB |
| `lightgbm_model.txt` | LightGBM | 99.43% | 2.18 mm | 17 MB |
| `catboost_model.cbm` | CatBoost | 99.10% | 2.74 mm | 48 MB |

## Support Files

- **scaler.pkl** - StandardScaler for preprocessing (essential for predictions)
- **feature_names.npy** - List of 175+ feature names
- **model_comparison_table.csv** - Complete metrics for all models
- **complete_model_comparison.png** - Comprehensive visualization dashboard
- **COMPREHENSIVE_RESULTS_REPORT.txt** - Detailed performance report

## Usage

```python
import torch
import joblib

# Load best model (Wide & Deep)
model = torch.load('models/widedeep_model.pt')
scaler = joblib.load('models/scaler.pkl')
feature_names = np.load('models/feature_names.npy', allow_pickle=True)

model.eval()

# Make predictions
X_scaled = scaler.transform(features)
with torch.no_grad():
    predictions = model(torch.FloatTensor(X_scaled)).numpy()
```

## Note on Large Files

- XGBoost (83 MB) and CatBoost (48 MB) models are large due to ensemble size
- Consider using Wide & Deep (354 KB) for deployment if size is a concern
- All models are production-ready and can be loaded directly
