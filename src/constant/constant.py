import os

# Path
LOG_DIR = "/media/ahmed/Data/0ps/log"
LOG_FILE = os.path.join(LOG_DIR, "logging.log")
RAW_DATA = "/media/ahmed/Data/0ps/data/raw/ai_adoption_dataset.csv"
REPORT1 = "/media/ahmed/Data/0ps/reports/report_1.txt"
DATA1 = "/media/ahmed/Data/0ps/data/processed/data1.csv"
SCALER_MODEL = "/media/ahmed/Data/0ps/models/scaler.joblib"
CLUSTER_MODEL = "/media/ahmed/Data/0ps/models/cluster.joblib"
MODEL_CLASSIFIER = "/media/ahmed/Data/0ps/models/classifier.keras"


# Logging configuration
LOGGING_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
DATEFMT = "%Y-%m-%d %H:%M:%S"

# Features Engineering
COMPANY_SIZE_MAP = {"Startup": 50, "SME": 500, "Enterprise": 5000}

NUMERICAL_COL = [
    "adoption_rate",
    "daily_active_users",
    "company_size_numeric",
    "user_engagement_rate",
]

FINAL_COLUMNS = [
    "country",
    "industry",
    "ai_tool",
    "adoption_rate",
    "daily_active_users",
    "age_group",
    "company_size",
    "company_size_numeric",
    "user_engagement_rate",
]

# Visualization
WIDTH = 1200
HEIGHT = 800

# Cluster
N_CLUSTER = 2
RANDOM_STATE = 0
N_INIT = 10

# Train Test Split
TEST_SIZE = 0.2


# Model
INPUT_SHAPE = (9,)
OUT_CLASSES = 5
ACTIVATION = "relu"
EPOCHS = 20
