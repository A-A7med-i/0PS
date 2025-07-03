# AI Tool Classification

This project analyzes and classifies the adoption of AI tools across industries and countries using a large dataset. It provides data loading, preprocessing, feature engineering, and reporting utilities to support exploratory data analysis and further machine learning tasks.

## Project Structure

```
AI-Tool-Classification/
├── data/
│   ├── raw/                # Raw datasets (e.g., ai_adoption_dataset.csv)
│   └── processed/          # Processed datasets
├── log/
│   └── logging.log         # Log files
├── notebooks/
│   └── exploratory.ipynb   # Jupyter notebooks for EDA and experiments
├── reports/
│   └── report_1.txt        # Generated data analysis reports
├── src/
│   ├── constant/           # Project constants and paths
│   ├── data/               # Data loading utilities
│   ├── feature_engineering/# Feature engineering scripts
│   ├── preprocessing/      # Data preprocessing scripts
│   └── utils/              # Helper functions
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # Project documentation
```

## Main Features

- **Data Loading:** Utilities to load and validate large CSV datasets.
- **Feature Engineering:** Functions to create new features (e.g., company size numeric, user engagement rate).
- **Reporting:** Automated generation of data analysis reports.
- **Logging:** Centralized logging for debugging and tracking.
- **Jupyter Notebooks:** For exploratory data analysis and prototyping.

## Getting Started

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Exploratory Analysis:**
   Open and run the notebook in `notebooks/exploratory.ipynb` to see data loading, feature engineering, and report generation in action.

3. **Project Scripts:**
   - Data loading: `src/data/load_data.py`
   - Feature engineering: `src/feature_engineering/features.py`
   - Helper functions: `src/utils/helper.py`

## Dataset

- The main dataset is located at `data/raw/ai_adoption_dataset.csv` and contains information on AI tool adoption rates, industries, countries, user demographics, and more.
