from typing import Tuple, Optional, Dict, Any
from src.constant.constant import *
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NeuralNetworkClassifier:
    """
    Deep neural network classifier for multi-class classification tasks.

    This class provides a complete interface for building, training, and evaluating
    a deep neural network using TensorFlow/Keras.
    """

    def __init__(self, classifier_name: str = "deep_neural_classifier"):
        """
        Initialize the neural network classifier.

        Args:
            classifier_name: Descriptive name identifier for the classifier
        """
        self.classifier_name = classifier_name

        self.neural_network = self._build_network_architecture()
        logger.info(
            f"Neural network classifier '{self.classifier_name}' initialized successfully"
        )

    def _build_network_architecture(self) -> tf.keras.Model:
        """
        Build the deep neural network architecture with multiple hidden layers.

        Returns:
            Constructed Keras sequential model
        """
        try:
            sequential_model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=INPUT_SHAPE),
                    tf.keras.layers.Dense(32, activation=ACTIVATION),
                    tf.keras.layers.Dense(64, activation=ACTIVATION),
                    tf.keras.layers.Dense(128, activation=ACTIVATION),
                    tf.keras.layers.Dense(32, activation=ACTIVATION),
                    tf.keras.layers.Dense(OUT_CLASSES, activation="softmax"),
                ],
                name=self.classifier_name,
            )
            logger.info("Neural network architecture constructed successfully")
            return sequential_model

        except Exception as network_build_error:
            logger.error(
                f"Error constructing network architecture: {str(network_build_error)}"
            )
            raise

    def compile_neural_network(
        self,
        optimization_algorithm: str = "adam",
        loss_function: str = "sparse_categorical_crossentropy",
        evaluation_metrics: list = ["accuracy"],
    ) -> None:
        """
        Compile the neural network with specified training parameters.

        Args:
            optimization_algorithm: Optimizer algorithm for gradient descent
            loss_function: Loss function for training
            evaluation_metrics: List of metrics to monitor during training
        """
        try:
            self.neural_network.compile(
                optimizer=optimization_algorithm,
                loss=loss_function,
                metrics=evaluation_metrics,
            )
            logger.info("Neural network compiled successfully")

        except Exception as compilation_error:
            logger.error(f"Error compiling neural network: {str(compilation_error)}")
            raise

    def train_classifier(
        self,
        training_features: np.ndarray,
        training_labels: np.ndarray,
        number_of_epochs: Optional[int] = EPOCHS,
    ) -> tf.keras.callbacks.History:
        """
        Train the neural network classifier on provided data.

        Args:
            training_features: Input features for training
            training_labels: Target labels for training
            number_of_epochs: Number of complete passes through training data

        Returns:
            Training history containing metrics and losses
        """
        try:
            logger.info(
                f"Starting classifier training for {number_of_epochs} epochs..."
            )

            self.training_history = self.neural_network.fit(
                training_features,
                training_labels,
                epochs=number_of_epochs,
            )

            logger.info("Classifier training completed successfully")
            return self.training_history

        except Exception as training_error:
            logger.error(f"Error during classifier training: {str(training_error)}")
            raise

    def evaluate_classifier_performance(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Evaluate the trained classifier on test dataset.

        Args:
            test_features: Input features for testing
            test_labels: True labels for testing

        Returns:
            Tuple containing (test_loss, test_accuracy)
        """
        try:
            logger.info("Evaluating classifier performance on test data...")

            evaluation_results = self.neural_network.evaluate(
                test_features,
                test_labels,
            )

            test_loss, test_accuracy = evaluation_results[0], evaluation_results[1]
            logger.info(
                f"Evaluation completed - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )

            return test_loss, test_accuracy

        except Exception as evaluation_error:
            logger.error(f"Error during classifier evaluation: {str(evaluation_error)}")
            raise

    def predict_class_probabilities(
        self, input_features: np.ndarray, prediction_batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate class probability predictions for input samples.

        Args:
            input_features: Input features for prediction
            prediction_batch_size: Batch size for prediction processing

        Returns:
            Array of class probabilities for each input sample
        """
        try:
            class_probabilities = self.neural_network.predict(
                input_features, batch_size=prediction_batch_size
            )
            return class_probabilities

        except Exception as prediction_error:
            logger.error(
                f"Error during probability prediction: {str(prediction_error)}"
            )
            raise

    def save_trained_classifier(self, model_file_path: str) -> None:
        """
        Save the trained classifier to disk storage.

        Args:
            model_file_path: File path where the model should be saved
        """
        try:
            self.neural_network.save(model_file_path)
            logger.info(f"Trained classifier saved to {model_file_path}")

        except Exception as save_error:
            logger.error(f"Error saving trained classifier: {str(save_error)}")
            raise
