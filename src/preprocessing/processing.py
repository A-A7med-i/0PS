from typing import Dict, Any
import pandas as pd


def map_series_values(
    categorical_series: pd.Series, value_mapping: Dict[Any, int]
) -> pd.Series:
    """
    Map categorical values in a Series to integer codes using a lookup dictionary.

    Transforms categorical values to their corresponding integer representations
    based on the provided mapping. Unmapped values are converted to NaN, allowing
    for identification of unseen categories during encoding.

    Args:
        categorical_series: Series containing categorical values to encode
        value_mapping: Dictionary mapping original categorical values to
                    integer codes (e.g., {'cat': 0, 'dog': 1})

    Returns:
        Series with categorical values replaced by their integer mappings.
        Unmapped values become NaN.

    Example:
        >>> series = pd.Series(['A', 'B', 'C', 'A'])
        >>> mapping = {'A': 0, 'B': 1, 'C': 2}
        >>> encoded = map_series_values(series, mapping)
        >>> # Returns: [0, 1, 2, 0]
    """
    return categorical_series.map(value_mapping)


def encode_categorical_columns(
    input_dataframe: pd.DataFrame, column_mappings: Dict[str, Dict[Any, int]]
) -> pd.DataFrame:
    """
    Encode categorical columns in DataFrame using predefined mapping dictionaries.

    Applies integer encoding to specified categorical columns based on provided
    mapping dictionaries. Only columns present in the mappings dictionary are
    encoded, allowing selective transformation of categorical features. The
    operation modifies the DataFrame in-place for memory efficiency.

    Args:
        input_dataframe: DataFrame containing categorical columns to encode.
                        Modified in-place during encoding process.
        column_mappings: Nested dictionary structure where outer keys are column
                        names and inner dictionaries map categorical values to
                        integer codes. Format: {'column_name': {'value': code}}

    Returns:
        Same DataFrame object with specified categorical columns encoded as integers.
        Columns not in mappings remain unchanged.

    Example:
        >>> df = pd.DataFrame({
        ...     'color': ['red', 'blue', 'green'],
        ...     'size': ['S', 'M', 'L'],
        ...     'price': [10, 20, 30]
        ... })
        >>> mappings = {
        ...     'color': {'red': 0, 'blue': 1, 'green': 2},
        ...     'size': {'S': 0, 'M': 1, 'L': 2}
        ... }
        >>> encoded_df = encode_categorical_columns(df, mappings)
        >>> # 'color' and 'size' columns now contain integer codes
    """
    for column_name in input_dataframe.columns:
        if column_name in column_mappings:
            input_dataframe.loc[:, column_name] = map_series_values(
                input_dataframe[column_name], column_mappings[column_name]
            )
    return input_dataframe
