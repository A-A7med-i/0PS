from src.model.model import NeuralNetworkClassifier
from src.data.load_data import LoadData
from typing import Tuple, Dict, Any
from src.constant.constant import *
from src.utils.helper import (
    create_categorical_mappings,
    save_dataframe_to_csv,
    persist_model_to_file,
    split_data,
)
from src.feature_engineering.features import (
    engineer_company_features,
    create_customer_clusters,
)
import pandas as pd
import numpy as np
import logging
import mlflow


logger = logging.getLogger(__name__)


class MLPipeline:
    """Improved ML Pipeline with better structure and error handling"""

    def __init__(self, experiment_name: str = "customer_classification"):
        self.experiment_name = experiment_name
        self.run_id = None
        self.artifacts = {}

    def setup_mlflow(self) -> None:
        """Setup MLflow experiment and run"""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment '{self.experiment_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate input data"""
        try:
            logger.info("Loading data...")
            load_data = LoadData()

            load_data.make_report(REPORT1)
            logger.info(f"Data quality report saved to {REPORT1}")

            logger.info(f"Data loaded successfully. Shape: {load_data.data.shape}")
            return load_data.data

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def feature_engineering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Perform feature engineering with proper error handling"""
        try:
            logger.info("Starting feature engineering...")

            engineered_data = engineer_company_features(data)
            logger.info("Company features engineered successfully")

            mapping = create_categorical_mappings(engineered_data)
            logger.info(f"Created mappings for {len(mapping)} categorical features")

            mlflow.log_param("original_features", data.shape[1])
            mlflow.log_param("engineered_features", engineered_data.shape[1])

            return engineered_data, mapping

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def create_clusters(
        self, data: pd.DataFrame, mapping: Dict
    ) -> Tuple[pd.DataFrame, Any, Any]:
        """Create customer clusters and save models"""
        try:
            logger.info("Creating customer clusters...")

            final_data, cluster_model, feature_scaler = create_customer_clusters(
                data, NUMERICAL_COL, mapping
            )

            persist_model_to_file(cluster_model, CLUSTER_MODEL)
            persist_model_to_file(feature_scaler, SCALER_MODEL)

            save_dataframe_to_csv(final_data, DATA1)

            mlflow.log_artifact(CLUSTER_MODEL, "models")
            mlflow.log_artifact(SCALER_MODEL, "models")
            mlflow.log_artifact(DATA1, "data")

            logger.info("Customer clustering completed successfully")
            return final_data, cluster_model, feature_scaler

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

    def prepare_training_data(
        self, data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data"""
        try:
            logger.info("Preparing training data...")

            data = pd.read_csv(data_path)
            x_train, x_test, y_train, y_test = split_data(data)

            x_train = np.array(x_train, dtype=np.float32)
            x_test = np.array(x_test, dtype=np.float32)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            mlflow.log_param("train_samples", len(x_train))
            mlflow.log_param("test_samples", len(x_test))
            mlflow.log_param("feature_dim", x_train.shape[1])
            mlflow.log_param("n_classes", len(np.unique(y_train)))

            logger.info(
                f"Training data prepared - Train: {x_train.shape}, Test: {x_test.shape}"
            )
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def train_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> NeuralNetworkClassifier:
        try:
            model = NeuralNetworkClassifier()
            model.compile_neural_network()

            model.train_classifier(x_train, y_train)

            logger.info("Model training completed successfully")
            return model

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def evaluate_model(
        self, model: NeuralNetworkClassifier, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate model performance"""
        try:
            logger.info("Evaluating model performance...")

            test_loss, test_accuracy = model.evaluate_classifier_performance(
                x_test, y_test
            )

            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)

            logger.info(
                f"Model evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}"
            )
            return test_loss, test_accuracy

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    def save_model(self, model: NeuralNetworkClassifier, model_path: str) -> None:
        """Save the trained model"""
        try:
            logger.info(f"Saving model to {model_path}")

            model.save_trained_classifier(model_path)

            mlflow.keras.log_model(model.neural_network, "model")

            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        try:
            self.setup_mlflow()

            with mlflow.start_run() as run:
                self.run_id = run.info.run_id
                logger.info(f"Started MLflow run: {self.run_id}")

                # Step 1: Load and validate data
                raw_data = self.load_and_validate_data()

                # Step 2: Feature engineering
                engineered_data, mapping = self.feature_engineering(raw_data)

                # Step 3: Create customer clusters
                final_data, cluster_model, feature_scaler = self.create_clusters(
                    engineered_data, mapping
                )

                # Step 4: Prepare training data
                x_train, x_test, y_train, y_test = self.prepare_training_data(DATA1)

                # Step 5: Train model
                model = self.train_model(x_train, y_train)

                # Step 6: Evaluate model
                test_loss, test_accuracy = self.evaluate_model(model, x_test, y_test)

                # Step 7: Save model
                self.save_model(model, MODEL_CLASSIFIER)

                logger.info("Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
