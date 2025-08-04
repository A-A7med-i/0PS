
# AI Tool Classification (MLOps Project)

**Description:**
MLOps project for analyzing, engineering features, and classifying AI tool adoption using reproducible pipelines. The repository demonstrates end-to-end machine learning workflows, including data versioning, experiment tracking, and modular code structure for scalability and collaboration.

This project provides:

- Automated data loading, preprocessing, and feature engineering
- Deep learning and clustering models for AI tool adoption analysis
- Reproducible pipelines with DVC and experiment tracking with MLflow
- Modular, extensible codebase for rapid experimentation and productionization

## Key Features

- **Modular Pipeline:** Data loading, preprocessing, feature engineering, clustering, and model training
- **Deep Learning:** Neural network classifier using TensorFlow/Keras
- **Clustering:** Unsupervised learning for customer or tool segmentation
- **MLOps Tools:** DVC for data versioning, MLflow for experiment tracking
- **Reproducibility:** All steps tracked and reproducible via DVC pipelines
- **Extensibility:** Easily add new pipeline stages or ML projects
- **Visualization:** Notebooks and scripts for EDA and result visualization

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw datasets (e.g., ai_adoption_dataset.csv)
│   └── processed/          # Processed datasets after cleaning/feature engineering
├── log/
│   └── logging.log         # Pipeline and application logs
├── models/                 # Saved models (e.g., model.keras, scaler.joblib)
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis notebook
│   ├── experiments.ipynb   # Experiment tracking notebook
│   └── exploratory.ipynb   # Main exploratory notebook
├── reports/
│   └── report_1.txt        # Generated data analysis/model reports
├── src/
│   ├── constant/           # Project-wide constants and configuration
│   ├── data/               # Data loading and validation utilities
│   ├── feature_engineering/# Feature engineering and transformation scripts
│   ├── model/              # Deep learning and ML model definitions
│   ├── pipeline/           # Pipeline orchestration and main scripts
│   ├── preprocessing/      # Data preprocessing and encoding scripts
│   ├── utils/              # Helper functions (saving/loading, splitting, scaling)
│   └── visualization/      # Visualization scripts
├── .dvcignore              # DVC ignore file
├── .gitignore              # Git ignore file
├── LICENSE                 # License file
├── requirements.txt        # Python dependencies
├── setup.py                # Python package setup
└── README.md               # Project documentation
```

## Getting Started

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up MLflow (for experiment tracking):**

   ```bash
   mlflow ui
   ```

3. **Run the pipeline:**

   ```bash
   python src/pipeline/main.py
   ```

4. **Explore notebooks:**
   Open any notebook in `notebooks/` (e.g., `exploratory.ipynb`, `EDA.ipynb`, or `experiments.ipynb`) for EDA and prototyping.

## Dataset

- Main dataset: `data/raw/ai_adoption_dataset.csv`
  Contains information on AI tool adoption rates, industries, countries, user demographics, and more.

## License

This project is licensed under the terms of the LICENSE file.

---

*For more details on each module or to contribute, please see the respective directories and code comments.*
