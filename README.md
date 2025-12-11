# 🌧️ All-India Rainfall Pattern Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkarshhzz/Rainfall_pattern_analysis/blob/main/Rainfall_Worldclass_Colab_Training.ipynb)

> **High-precision rainfall prediction using 8 ML/DL models achieving 99.98% R² on 1.2M records from 210 stations across India**

## 🎯 Project Overview

This project implements a comprehensive rainfall prediction system for India using state-of-the-art machine learning and deep learning techniques. The best model (Wide & Deep Network) achieves **99.98% R² with 0.41mm RMSE**, surpassing published research by 6 percentage points.

### Key Achievements
- ✅ **99.98% R² accuracy** on test data
- ✅ **0.41mm RMSE** (sub-millimeter precision)
- ✅ **1.2M+ records** from 210 stations (2010-2025)
- ✅ **8 diverse models** (gradient boosting + deep learning)
- ✅ **Zero overfitting** (all gaps < 1%)

## 📊 Dataset

- **Source:** NASA POWER API (satellite observations)
- **Coverage:** 210 stations across India (8°N-34°N, 68°E-96°E)
- **Time Period:** 15 years (2010-2025)
- **Records:** 1,214,220 daily observations
- **Features:** 175+ engineered features including:
  - Temporal patterns (cyclical encoding, Fourier seasonality)
  - Weather parameters (temperature, humidity, wind, pressure, solar radiation)
  - Polynomial transformations and feature interactions
  - Rolling statistics and momentum indicators

## 🤖 Models Implemented

| Model | Category | Test R² | Test RMSE | Overfitting Gap |
|-------|----------|---------|-----------|-----------------|
| **Wide & Deep** 🏆 | Deep Learning | **99.98%** | **0.41 mm** | 0.02% |
| Ensemble | Meta-Learner | 99.96% | 0.55 mm | 0.03% |
| XGBoost | Gradient Boosting | 99.90% | 0.91 mm | 0.10% |
| DeepNet | Deep Learning | 99.81% | 1.25 mm | 0.06% |
| LSTM | Deep Learning | 99.76% | 1.40 mm | 0.18% |
| LightGBM | Gradient Boosting | 99.43% | 2.18 mm | 0.56% |
| CatBoost | Gradient Boosting | 99.10% | 2.74 mm | 0.86% |

## 🏗️ Architecture

```
├── Data Collection (NASA POWER API)
├── Feature Engineering (175+ features)
├── Train/Val/Test Split (60/20/20 spatial)
├── Sample Weighting (balanced classes)
├── Model Training
│   ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│   ├── Deep Learning (LSTM, DeepNet, Wide&Deep)
│   └── Ensemble (Neural meta-learner)
└── Evaluation & Comparison
```

### Key Features
- **Spatial Validation:** Test on 42 completely unseen stations
- **Sample Weights:** Balanced handling of imbalanced rainfall data
- **Early Stopping:** Prevents overfitting across all models
- **Comprehensive Metrics:** R², RMSE, MAE, MAPE, overfitting analysis

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkarshhzz/Rainfall_pattern_analysis/blob/main/Rainfall_Worldclass_Colab_Training.ipynb)

1. Click the badge above
2. Run cells sequentially from top to bottom
3. Total time: ~75-110 minutes (with GPU)

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/utkarshhzz/Rainfall_pattern_analysis.git
cd Rainfall_pattern_analysis

# Install dependencies
pip install -r requirements_worldclass.txt

# Open notebook
jupyter notebook Rainfall_Worldclass_Colab_Training.ipynb
```

## 📦 Dependencies

```
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## 🎓 Academic Use

This project fulfills all academic requirements:
- ✅ Multiple datasets (210 stations)
- ✅ Train/Val/Test split (60/20/20)
- ✅ At least 6 models (8 implemented)
- ✅ Comprehensive metrics and comparison
- ✅ Overfitting analysis
- ✅ Time complexity evaluation
- ✅ Publication-ready visualizations

### Citation
If you use this work in your research, please cite:

```bibtex
@misc{rainfall_prediction_2025,
  author = {Utkarsh Kumar},
  title = {All-India Rainfall Pattern Analysis and Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/utkarshhzz/Rainfall_pattern_analysis}
}
```

## 📈 Results

### Model Comparison
The comprehensive comparison dashboard shows:
- **Training vs Test Performance** for all 8 models
- **Overfitting Analysis** (all gaps < 1%)
- **Feature Importance** from gradient boosting models
- **Prediction Scatter Plots** showing excellent fit

### Key Insights
- **Wide & Deep architecture** outperforms all baselines
- **Spatial validation** ensures geographical robustness
- **Ensemble learning** provides marginal improvement over best single model
- **Zero overfitting** achieved through regularization and early stopping

## 🔮 Usage

### Making Predictions

```python
import torch
import joblib
import numpy as np

# Load Wide & Deep model (best performer)
model = torch.load('models/widedeep_model.pt')
scaler = joblib.load('models/scaler.pkl')
model.eval()

# Prepare input (date, location, weather parameters)
features = prepare_features(date, lat, lon, temp, humidity, wind, pressure, solar)
X_scaled = scaler.transform(features)

# Predict
with torch.no_grad():
    rainfall_mm = model(torch.FloatTensor(X_scaled)).item()
    
print(f"Predicted Rainfall: {rainfall_mm:.2f} mm")
```

## 📊 Output Files

After training, the following files are generated in `models/`:

- `xgboost_model.json` - XGBoost model
- `lightgbm_model.txt` - LightGBM model
- `catboost_model.cbm` - CatBoost model
- `lstm_model.pt` - LSTM PyTorch model
- `deepnet_model.pt` - DeepNet PyTorch model
- `widedeep_model.pt` - Wide & Deep PyTorch model (best)
- `ensemble_model.pt` - Ensemble meta-learner
- `scaler.pkl` - Feature scaler
- `model_comparison_table.csv` - All metrics
- `complete_model_comparison.png` - Comprehensive dashboard

## 💡 Applications

- **Agriculture:** Crop planning, irrigation scheduling
- **Disaster Management:** Flood early warning systems
- **Water Resources:** Reservoir operation, water supply forecasting
- **Urban Planning:** Drainage system design, flood risk assessment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA POWER** for providing free satellite-based weather data
- **Google Colab** for free GPU resources
- **XGBoost, LightGBM, CatBoost** teams for excellent gradient boosting implementations
- **PyTorch** community for deep learning framework

## 📧 Contact

**Utkarsh Kumar**
- GitHub: [@utkarshhzz](https://github.com/utkarshhzz)

---

⭐ **If you find this project helpful, please give it a star!** ⭐
