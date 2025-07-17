from src.constant.constant import RAW_DATA
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import datetime
import logging
import io


logger = logging.getLogger(__name__)


class LoadData:
    """
    A class to load and analyze CSV data files.

    Attributes:
        path (Union[str, Path]): Path to the CSV file
        data (Optional[pd.DataFrame]): Loaded DataFrame
    """

    def __init__(self, path: Union[str, Path] = RAW_DATA):
        """
        Initialize LoadData with file path.

        Args:
            path: Path to the CSV file (default: RAW_DATA constant)
        """
        self.path = Path(path)
        self.data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        try:
            self.data = pd.read_csv(self.path)
            logger.info(f"Successfully loaded data from {self.path}")
            logger.info(f"Data shape: {self.data.shape}")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

    def get_information(self) -> str:
        """
        Get comprehensive information about the dataset.

        Returns:
            str: Formatted string containing dataset info and description
        """
        try:
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            info_str = buffer.getvalue()

            description = self.data.describe().T

            information = f"""
╭{'─' * 78}╮
│{' ' * 26}ℹ️ DATASET INFO{' ' * 27}│
╰{'─' * 78}╯
{info_str}

╭{'─' * 78}╮
│{' ' * 22}📈 STATISTICAL SUMMARY{' ' * 22}│
╰{'─' * 78}╯
{description.to_string()}

╭{'─' * 78}╮
│{' ' * 24}📋 DATASET DETAILS{' ' * 24}│
╰{'─' * 78}╯
📁 File Path      : {self.path}
📐 Shape          : {self.data.shape[0]:,} rows × {self.data.shape[1]:,} columns
💾 Memory Usage   : {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB

╭{'─' * 78}╮
│{' ' * 22}⚠️  DATA QUALITY CHECK{' ' * 22}│
╰{'─' * 78}╯
Missing Values per Column:
{self.data.isnull().sum().to_string()}

{'═' * 80}
"""

            return information

        except Exception as e:
            logger.error(f"Error generating information: {e}")
            return f"Error generating dataset information: {str(e)}"

    def get_basic_info(self) -> dict:
        """
        Get basic information as a dictionary.

        Returns:
            dict: Dictionary containing basic dataset information
        """

        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "file_path": str(self.path),
        }

    def make_report(
        self,
        path: Union[str, Path],
    ) -> bool:
        """
        Generate a comprehensive data report and save it to a text file.

        Args:
            path: Output file path
            include_sample_data: Whether to include sample data in the report

        Returns:
            bool: True if report was created successfully, False otherwise
        """
        try:
            output_path = Path(path)
            information = self.get_information()
            basic_info = self.get_basic_info()

            report_content = self._generate_text_report(information, basic_info)

            with open(path, "w") as filename:
                filename.write(report_content)

            print(f"Content successfully written to '{output_path}'")
            return True

        except Exception as e:
            error_msg = f"Unexpected error generating report: {e}"
            logger.error(error_msg)
            print(error_msg)
            return False

    def _generate_text_report(
        self,
        information: str,
        basic_info: dict,
    ) -> str:
        """Generate a text format report."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        column_names_formatted = "\n".join(
            f"  - {col}" for col in basic_info["columns"]
        )

        dtypes_formatted = "\n".join(
            f"  {col}: {dtype}" for col, dtype in basic_info["dtypes"].items()
        )

        missing_values_data = [
            f"  {col}: {count}"
            for col, count in basic_info["missing_values"].items()
            if count > 0
        ]

        if missing_values_data:
            missing_values_formatted = "\n".join(missing_values_data)
        else:
            missing_values_formatted = "  No missing values detected."

        report = f"""
╔{'═' * 78}╗
║{' ' * 25}📊 DATA ANALYSIS REPORT{' ' * 25}║
║{' ' * 20}Generated: {timestamp}{' ' * (58 - len(timestamp))}║
╚{'═' * 78}╝

{information.strip()}

╭{'─' * 78}╮
│{' ' * 22}📋 DATASET OVERVIEW{' ' * 22}│
╰{'─' * 78}╯

🔢 Dimensions    : {basic_info['shape'][0]:,} rows × {basic_info['shape'][1]:,} columns
📊 Total Columns : {len(basic_info['columns']):,}
💾 Memory Usage  : {basic_info['memory_usage_mb']:.2f} MB
📁 File Location : {basic_info['file_path']}

╭{'─' * 78}╮
│{' ' * 28}📝 COLUMNS{' ' * 28}│
╰{'─' * 78}╯
{column_names_formatted}

╭{'─' * 78}╮
│{' ' * 26}🏷️  DATA TYPES{' ' * 26}│
╰{'─' * 78}╯
{dtypes_formatted}

╭{'─' * 78}╮
│{' ' * 24}⚠️  MISSING VALUES{' ' * 24}│
╰{'─' * 78}╯
{missing_values_formatted}

{'═' * 80}
"""

        return report

    def get_column_info(self, column: str) -> dict:
        """
        Get detailed information about a specific column.

        Args:
            column: Column name

        Returns:
            dict: Column information
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataset")

        col_data = self.data[column]

        info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "non_null_count": col_data.count(),
            "null_count": col_data.isnull().sum(),
            "unique_count": col_data.nunique(),
            "memory_usage": col_data.memory_usage(deep=True),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            info.update(
                {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "median": col_data.median(),
                }
            )

        if col_data.dtype == "object" or pd.api.types.CategoricalDtype(col_data):
            info["top_values"] = col_data.value_counts().head().to_dict()

        return info

    def reload_data(self, new_path: Optional[Union[str, Path]] = None) -> None:
        """
        Reload data from file.

        Args:
            new_path: Optional new path to load data from
        """
        if new_path:
            new_path = Path(new_path)

        self._load_data()
