
# Statistical Analysis of Premier League Players for Fantasy Football Performance Prediction

## Description
This project analyses and trains a classification model to predict Fantasy Premier League (FPL) points.

## Getting Started

### Setting Up the Virtual Environment
To install the virtual environment, follow these steps:

1. Create the virtual environment:
   ```
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Downloading the Data

### Automated Download and Preprocessing
To download and preprocess the data (requires Kaggle API key), run:
```
python src/data_preprocessing.py
```

### Manual Download
Data can also be directly downloaded from the following links:
- [2022/2023 Football Player Stats](https://www.kaggle.com/datasets/vivovinco/20222023-football-player-stats)
- [Fantasy Premier League Dataset 2022/2023](https://www.kaggle.com/datasets/meraxes10/fantasy-premier-league-dataset-2022-2023)

The downloaded data must be saved in `data/raw/`.

If data is downloaded directly, run:
```
src/data_preprocessing.py --download_data False
```

### Training and Testing the Model
To train and test the model, run:
```
python main.py --train_model xgboost --feature_selection_model rfe --num_features 30  
```
