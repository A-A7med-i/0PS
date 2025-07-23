from src.constant.constant import WIDTH, HEIGHT
from plotly.express.colors import qualitative
import plotly.graph_objects as go
import pandas as pd
import random


COLORS = qualitative.Set3 + qualitative.Set1 + qualitative.Vivid


def create_box_plot(dataframe: pd.DataFrame, column_name: str) -> go.Figure:
    """
    Generates a box plot for a specified numeric column in a Pandas DataFrame.

    This function creates a Plotly box plot to visualize the distribution of a
    numeric column, showing the median, quartiles, and potential outliers.
    The mean is also indicated on the plot.

    Args:
        dataframe (pd.DataFrame): The input Pandas DataFrame containing the data.
        column_name (str): The name of the numeric column to plot.

    Returns:
        go.Figure: A Plotly Figure object representing the box plot.

    Raises:
        ValueError: If the specified column_name is not found in the DataFrame
                    or if the column is not numeric.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
        raise ValueError(f"Column '{column_name}' must be numeric for a box plot.")

    box_data = go.Box(
        x=dataframe[column_name],
        boxmean=True,
        name=column_name,
        marker_color=random.choice(COLORS),
    )

    plot_layout = go.Layout(
        title={
            "text": f"Distribution of {column_name}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24},
        },
        xaxis_title=f"{column_name}",
        yaxis_title="Value",
        showlegend=False,
        plot_bgcolor="white",
        width=WIDTH,
        height=HEIGHT,
    )

    figure = go.Figure(data=[box_data], layout=plot_layout)

    return figure


def create_bar_plot(dataframe: pd.DataFrame, column_name: str) -> go.Figure:
    """
    Generates a bar plot for the distribution of a categorical or object column in a Pandas DataFrame.

    This function calculates the value counts of the specified column and
    creates a Plotly bar chart to visualize their distribution.

    Args:
        dataframe (pd.DataFrame): The input Pandas DataFrame containing the data.
        column_name (str): The name of the categorical or object column to plot.

    Returns:
        go.Figure: A Plotly Figure object representing the bar plot.

    Raises:
        ValueError: If the specified column_name is not found in the DataFrame.
        TypeError: If the specified column is not of 'object' (string) or 'category' dtype.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if not (
        dataframe[column_name].dtype == "object"
        or pd.api.types.is_categorical_dtype(dataframe[column_name])
    ):
        raise TypeError(
            f"Column '{column_name}' must be of 'object' (string) or 'category' dtype for a bar plot."
        )

    category_counts = dataframe[column_name].value_counts()

    bar_trace = go.Bar(
        x=category_counts.index,
        y=category_counts.values,
        marker_color=COLORS,
        opacity=0.8,
        name="Category Counts",
    )

    plot_layout = go.Layout(
        title_text=f"Distribution of {column_name}",
        title_x=0.5,
        title_font={"size": 24},
        template="plotly_white",
        xaxis_title=f"{column_name} Categories",
        yaxis_title="Count",
        bargap=0.2,
        showlegend=False,
        width=WIDTH,
        height=HEIGHT,
    )

    figure = go.Figure(data=[bar_trace], layout=plot_layout)

    return figure


def visualize_correlation_heatmap(dataset: pd.DataFrame) -> None:
    """
    Generate and display an interactive correlation heatmap for numerical features.

    Creates a correlation matrix from the input DataFrame and visualizes it as
    an interactive heatmap using Plotly. The heatmap displays correlation
    coefficients rounded to 2 decimal places with a diverging color scale
    where strong positive correlations appear in red and strong negative
    correlations appear in blue.

    Args:
        dataset: DataFrame containing numerical columns for correlation analysis.
                Only numerical columns will be included in the correlation matrix.

    Returns:
        None. Displays the interactive heatmap plot directly.

    Note:
        - Correlation coefficients are rounded to 2 decimal places for readability
        - Uses RdBu_r (Red-Blue reversed) colorscale for intuitive interpretation
        - Plot dimensions are controlled by global WIDTH and HEIGHT constants
        - Requires numerical data; categorical columns are automatically excluded
    """
    correlation_matrix = dataset.corr().round(2)

    correlation_figure = go.Figure(
        go.Heatmap(
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            z=correlation_matrix.values,
            text=correlation_matrix.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="RdBu_r",
        )
    )

    correlation_figure.update_layout(
        title=dict(text="Correlation Heatmap", x=0.5), width=WIDTH, height=HEIGHT
    )

    correlation_figure.show()
