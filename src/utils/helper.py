from sklearn.model_selection import train_test_split
from typing import Any, Union, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from src.constant.constant import TEST_SIZE
from plotly import graph_objects as go
from pathlib import Path
import pandas as pd
import logging
import joblib
import json
import os


logger = logging.getLogger(__name__)


def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str):
    """
    Saves a Pandas DataFrame to a CSV file.

    This function takes a DataFrame and a file path, then saves the DataFrame
    to the specified path in CSV format. It ensures that the directory
    for the file exists before saving.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The full path to the output CSV file (e.g., "data/processed/output.csv").

    Raises:
        TypeError: If the input 'dataframe' is not a pandas DataFrame or 'file_path' is not a string.
        IOError: If there's an issue writing the file to the specified path.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
    if not isinstance(file_path, str):
        raise TypeError("Input 'file_path' must be a string.")

    output_path = Path(file_path)

    try:
        dataframe.to_csv(output_path, index=False)
        logger.info(f"DataFrame successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {output_path}: {e}")
        raise


def save_plotly_figure(figure: go.Figure, output_path: str) -> bool:
    """
    Saves a Plotly figure to a specified file path.

    This function supports saving figures as HTML files (for interactive plots)
    or as static image files (e.g., PNG, JPEG, SVG) based on the file extension
    provided in the output_path. It also ensures that the directory for the
    output_path exists.

    Args:
        figure (go.Figure): The Plotly figure object to be saved.
        output_path (str): The full path including the filename and extension
                        where the figure will be saved. Supported extensions
                        include '.html' for interactive plots, and various
                        image formats (e.g., '.png', '.jpeg', '.svg') for static images.

    Returns:
        bool: True if the figure was saved successfully, False otherwise.

    Raises:
        ValueError: If the 'figure' object is None or if 'output_path' is empty.
        Exception: Catches and logs any other exceptions that occur during the
                saving process, re-raising them for upstream handling.
    """
    try:
        if figure is None:
            raise ValueError("The Plotly figure object cannot be None.")

        if not output_path:
            raise ValueError("The output_path cannot be empty.")

        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        if output_path.lower().endswith(".html"):
            figure.write_html(output_path)
        else:
            figure.write_image(output_path)

        logger.info(f"Figure saved successfully to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save figure to {output_path}: {e}")
        raise


def create_categorical_mappings(data: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
    """
    Creates integer mappings for categorical and object columns in a DataFrame.

    Each unique value in a qualifying column is assigned a unique integer ID,
    starting from 0, in alphabetical order of the unique values.

    Args:
        data: The input DataFrame.

    Returns:
        A dictionary where keys are column names (str) and values are
        dictionaries. Each inner dictionary maps unique column values (Any)
        to their corresponding integer IDs (int).
    """
    return {
        col: {value: idx for idx, value in enumerate(sorted(data[col].unique()))}
        for col in data.select_dtypes(include=["object", "category"]).columns
    }


def save_data_as_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Saves serializable data to a JSON file with pretty-printing.

    The parent directories for the given file path will be created if they
    do not already exist.

    Args:
        data: The data to be saved. Must be JSON-serializable.
        file_path: The full path to the JSON file, including the filename and
                extension (e.g., "data/config.json").

    Raises:
        TypeError: If the provided `data` is not JSON serializable.
        OSError: If there is an issue writing the file to the specified path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_data_from_json(file_path: Union[str, Path]) -> Any:
    """
    Loads data from a JSON file.

    Args:
        file_path: The full path to the JSON file, including the filename and
                extension (e.g., "data/config.json").

    Returns:
        The loaded data, which can be of any type that was originally
        serialized into the JSON file (e.g., dict, list, str, int, float, bool, None).

    Raises:
        FileNotFoundError: If the specified `file_path` does not exist.
        json.JSONDecodeError: If the file at `file_path` contains invalid JSON.
        OSError: If there is an issue reading the file from the specified path.
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def scale_features(
    dataframe: pd.DataFrame, features_to_scale: Union[str, List[str]]
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scales specified features in a DataFrame using StandardScaler.

    This function applies Z-score normalization (standard scaling) to one or
    more columns, transforming them to have a mean of 0 and a standard
    deviation of 1. The original DataFrame's structure and any non-scaled
    columns are preserved.

    Args:
        dataframe: The input pandas DataFrame containing the features to be scaled.
        features_to_scale: The name of the column (as a string) or a list of
            column names to scale.

    Returns:
        A tuple containing:
        - pd.DataFrame: A new DataFrame with the specified features scaled.
        - StandardScaler: The fitted `StandardScaler` instance, which can be
        used to apply the same transformation to new data (e.g., a test set).

    Raises:
        ValueError: If the input DataFrame is empty.
        KeyError: If any of the column names in `features_to_scale` are not
            found in the DataFrame.
    """
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty")

    scaled_dataframe = dataframe.copy()

    feature_scaler = StandardScaler()

    feature_scaler.set_output(transform="pandas")

    scaled_features = feature_scaler.fit_transform(scaled_dataframe[features_to_scale])

    scaled_dataframe[features_to_scale] = scaled_features

    return scaled_dataframe, feature_scaler


def persist_model_to_file(trained_model: Any, output_filepath: str) -> None:
    """
    Persist a trained machine learning model to disk using joblib serialization.

    Serializes and saves the provided model object to the specified file location
    using joblib, which is optimized for scikit-learn models and NumPy arrays.
    Provides comprehensive error handling for common file system issues and
    logs successful operations for audit trails.

    Args:
        trained_model: The fitted machine learning model to serialize and save.
                    Compatible with scikit-learn estimators, clustering models,
                    preprocessors, and other joblib-serializable objects.
        output_filepath: Complete file path including directory, filename, and
                        extension where the model will be saved. Recommended
                        extensions: '.joblib', '.pkl', or '.pickle'
                        (e.g., 'models/kmeans_clustering_v1.joblib')

    Raises:
        IOError: When file system operations fail due to invalid paths,
                insufficient permissions, disk space limitations, or
                directory access issues. Includes original error context.

    Example:
        >>> from sklearn.cluster import KMeans
        >>> model = KMeans(n_clusters=3).fit(training_data)
        >>> persist_model_to_file(model, 'saved_models/customer_segments.joblib')
        >>> # Logs: "Model successfully persisted to: 'saved_models/customer_segments.joblib'"
    """
    try:
        joblib.dump(trained_model, output_filepath)
        logger.info(f"Model successfully persisted to: '{output_filepath}'")

    except OSError as filesystem_error:
        raise IOError(
            f"Failed to persist model to '{output_filepath}'. "
            f"Verify path exists, check write permissions, and ensure sufficient disk space. "
            f"Original error: {filesystem_error}"
        ) from filesystem_error


def load_model_from_file(input_filepath: str) -> Any:
    """
    Load a serialized machine learning model from disk using joblib deserialization.

    Deserializes and loads a previously saved model object from the specified file
    location using joblib. Provides comprehensive error handling for common file
    system issues and model compatibility problems, with detailed logging for
    successful operations and debugging support.

    Args:
        input_filepath: Complete file path to the saved model file including
                    directory, filename, and extension. Should match the path
                    used when saving the model (e.g., 'models/kmeans_clustering_v1.joblib')

    Returns:
        The deserialized machine learning model object ready for inference.
        Return type depends on the original model type (e.g., KMeans,
        RandomForestClassifier, StandardScaler, etc.)

    Raises:
        FileNotFoundError: When the specified file path does not exist or
                        cannot be accessed due to permission restrictions.
                        Includes suggestions for path verification.

    Example:
        >>> loaded_kmeans = load_model_from_file('saved_models/customer_segments.joblib')
        >>> # Logs: "Model successfully loaded from: 'saved_models/customer_segments.joblib'"
        >>> predictions = loaded_kmeans.predict(new_data)
    """
    try:
        loaded_model = joblib.load(input_filepath)
        logger.info(f"Model successfully loaded from: '{input_filepath}'")
        return loaded_model

    except FileNotFoundError as file_not_found_error:
        raise FileNotFoundError(
            f"Model file not found at '{input_filepath}'. "
            f"Verify the file path exists and check file permissions. "
            f"Original error: {file_not_found_error}"
        ) from file_not_found_error


def split_data(
    data: pd.DataFrame,
    target_column: str = "ai_tool",
    test_size: float = TEST_SIZE,
    random_state: int = 42,
    stratify_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits a DataFrame into features and a target, then into train/test sets.

    This utility function separates a DataFrame into features (X) and a target
    variable (y), then performs a train-test split. It is designed to be
    reproducible and supports stratified splitting for classification tasks.

    Args:
        data: The input pandas DataFrame to be split.
        target_column: The name of the column to be used as the target (y).
        test_size: The proportion of the dataset to allocate to the test set.
        random_state: Seed for the random number generator to ensure the split
            is always the same.
        stratify_split: If True, performs a stratified split based on the
            target variable, preserving class proportions in the train and test sets.

    Returns:
        A tuple containing four pandas objects in the standard order:
        (X_train, X_test, y_train, y_test)

    Raises:
        KeyError: If the `target_column` is not present in the DataFrame.
    """

    X = data.drop(columns=[target_column])
    y = data[target_column]

    stratify_option = y if stratify_split else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_option,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test
