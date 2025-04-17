# Financial Prediction Model

This project implements a machine learning system for financial prediction using multiple deep learning architectures. The system processes financial data, trains different models (MLP, LSTM, and Transformer), and provides comprehensive model interpretability analysis.

## Project Structure

```
.
├── FidelFolio_Dataset.csv      # Raw input dataset
├── preprocess_data_v2.py       # Data preprocessing script
├── main_contents.py            # Main model training and evaluation script
├── results/                    # Model evaluation results
├── plots/                      # Performance visualization plots
├── models/                     # Saved model files
└── interpretability/           # Model interpretability analysis
```

## Features

- **Data Preprocessing**
  - Handles missing values using KNN imputation
  - Outlier detection and handling using IQR method
  - Feature normalization using StandardScaler
  - Target-specific dataset creation
  - Time series sequence preparation

- **Model Architectures**
  - Multi-Layer Perceptron (MLP)
  - Long Short-Term Memory (LSTM)
  - Transformer with self-attention

- **Training Features**
  - Cross-validation with strict time-based splits
  - Early stopping and learning rate reduction
  - Model checkpointing for best performers

- **Model Interpretability**
  - Feature importance analysis
  - SHAP (SHapley Additive exPlanations) values
  - Prediction confidence visualization
  - Error distribution analysis
  - Cross-target feature importance comparison

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
This will create three preprocessed datasets, one for each target variable.

2. Train and evaluate models:
```bash
python main_contents.py
```
This will:
- Train MLP, LSTM, and Transformer models for each target
- Generate performance metrics and visualizations
- Create interpretability analysis
- Save results in respective directories

## Results

The results are organized in the following directories:

- `results/`: Contains CSV files with detailed prediction results
- `plots/`: Contains performance visualization plots
- `models/`: Contains saved model files
- `interpretability/`: Contains feature importance analysis and visualizations

## Model Performance

The system evaluates models using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Cross-validation performance
- Feature importance rankings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
