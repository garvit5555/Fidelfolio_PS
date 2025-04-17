# Financial Market Growth Prediction

This project implements a sophisticated machine learning system for predicting market capitalization growth of companies using historical financial indicators. The system employs multiple deep learning architectures and provides comprehensive model interpretability analysis.

## Dataset Description

The `FidelFolio_Dataset.csv` contains financial indicators for companies with the following structure:
- **Year**: Fiscal year of the data (2001-2023)
- **Company**: Company identifier
- **Features (1-28)**: Financial indicators and ratios
- **Targets**:
  - Target 1: 1-year forward market cap growth (%)
  - Target 2: 2-year forward market cap growth (%)
  - Target 3: 3-year forward market cap growth (%)

## Project Structure

```
.
├── FidelFolio_Dataset.csv      # Raw financial dataset
├── preprocess_data_v2.py       # Data preprocessing pipeline
├── main_contents.py            # Model training and evaluation
├── results/                    # Prediction results and metrics
├── plots/                      # Visualization outputs
├── models/                     # Saved model checkpoints
└── interpretability/           # Model interpretation analysis
```

## Technical Implementation

### Data Preprocessing (`preprocess_data_v2.py`)
1. **Data Cleaning**
   - Handles missing values using KNN imputation (n_neighbors=5)
   - Removes outliers using IQR method with winsorization
   - Standardizes features using StandardScaler

2. **Target-Specific Processing**
   - Creates separate datasets for each prediction horizon
   - Maintains data integrity by filtering null targets
   - Performs temporal alignment for time series creation

### Model Architecture (`main_contents.py`)

1. **Multi-Layer Perceptron (MLP)**
   ```python
   - Input Layer
   - Dense(64) + BatchNorm + Dropout(0.3)
   - Dense(32) + Dropout(0.2)
   - Dense(1) [Output]
   ```

2. **LSTM Network**
   ```python
   - Input Layer
   - LSTM(32, return_sequences=True)
   - Dropout(0.3)
   - LSTM(16)
   - Dropout(0.2)
   - Dense(1) [Output]
   ```

3. **Transformer**
   ```python
   - Input Layer
   - MultiHeadAttention(heads=4, key_dim=8)
   - LayerNormalization
   - Dense(32) + Dense(feature_dim)
   - LayerNormalization
   - Dense(1) [Output]
   ```

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error
- **Batch Size**: 32
- **Early Stopping**: Patience=10, monitor='val_loss'
- **Learning Rate Reduction**: Factor=0.5, patience=5, min_lr=1e-6

### Model Interpretability
1. **Feature Importance Analysis**
   - Permutation importance calculation
   - SHAP (SHapley Additive exPlanations) values
   - Cross-target feature importance comparison

2. **Prediction Analysis**
   - Confidence visualization with error mapping
   - Error distribution analysis
   - Performance comparison across targets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess the data:
```bash
python preprocess_data_v2.py
```
This creates three preprocessed datasets:
- `processed_data_target_1.csv`: 1-year prediction dataset
- `processed_data_target_2.csv`: 2-year prediction dataset
- `processed_data_target_3.csv`: 3-year prediction dataset

2. Train and evaluate models:
```bash
python main_contents.py
```

### Output Structure
- **results/**
  - `model_comparison_results.csv`: Overall performance metrics
  - `{model_type}_target_{n}_predictions.csv`: Detailed predictions

- **plots/**
  - Model performance comparisons
  - Learning curves
  - Error distributions

- **interpretability/**
  - Feature importance rankings
  - SHAP value analysis
  - Cross-target comparisons

## Model Performance Metrics

The system evaluates models using:
- **MSE (Mean Squared Error)**: Primary training objective
- **RMSE (Root Mean Squared Error)**: Interpretable error metric
- **Cross-validation Performance**: Using strict time-based splits
- **Feature Importance Rankings**: For model interpretability

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{fidelfolio_prediction,
  title = {Financial Market Growth Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/repository}
}
``` 
