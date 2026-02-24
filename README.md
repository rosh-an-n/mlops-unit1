# mlops-unit1

Basic ML project for MLOps Unit 1 assignment.

## Project Structure

```
mlops-unit1/
├── data/
├── src/
│   ├── data_stats.py
│   └── train.py
├── models/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo:
```bash
git clone https://github.com/your-username/mlops-unit1.git
cd mlops-unit1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Print dataset statistics:
```bash
python src/data_stats.py
```

### Train the model:
```bash
python src/train.py
```

This will train a Logistic Regression model on the Iris dataset and save it to `models/model.pkl`.

## What it does

- Loads the Iris dataset
- Splits into 80/20 train/test
- Trains a Logistic Regression model
- Prints accuracy
- Saves the model using joblib
