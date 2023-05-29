import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class Starts:
    """
    Attributes:
    -----------
    None

    Methods:
    --------
    jupyter_settings(palette: list[str]) -> None:
    Sets Jupyter Notebook settings such as figure size, font scale, color palette, etc.
    
    data_dimensions(data: pd.DataFrame) -> pd.DataFrame:
    Prints the number of rows and columns of a given Pandas DataFrame.

    type_na(data: pd.DataFrame) -> pd.DataFrame:
    A dataframe containing the data type, number of missing values and percentage of missing values for each column in the input dataframe.

    statistics_info( data: pd.DataFrame) -> pd.DataFrame:
    Computes basic statistics for numerical columns in a pandas DataFrame.

    plot_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    Plot countplots for all categorical features in the given dataframe.
    """
    def __init__(self):
        pass

    def jupyter_settings(self, palette= ["#00BFFF", "#DC143C", "#FFA07A", "#FFD700", "#8B008B", "#7CFC00", "#FF69B4", "#00CED1", "#FF6347", "#9400D3"]):
        """
        Sets Jupyter Notebook settings such as figure size, font scale, color palette, etc.

        Parameters:
        -----------
        palette : list[str], optional
            A list of hex codes representing colors to use in plots. Default is a list of 10 colors.

        Returns:
        --------
        None
        """
        plt.style.use('bmh')
        plt.rcParams.update({
            'figure.figsize': [24, 12],
            'font.size': 10
        })
        pd.options.display.max_columns = None
        pd.options.display.max_rows = 50
        pd.set_option('display.expand_frame_repr', False)
        sns.set(font_scale=2, palette=palette)
        sns.set_theme()
        warnings.simplefilter("ignore")
        sns.set_style("white")
        
        return None
    
    def data_dimensions( self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prints the number of rows and columns of a given Pandas DataFrame.
        
        Args:
        - data (pd.DataFrame): the DataFrame to check dimensions of
        
        Returns:
        - None
        """

        print( f'Number of rows: {data.shape[0]}')
        print( f'Number of cols: {data.shape[1]}')

        return None


    def type_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function takes a pandas dataframe as input and returns a dataframe containing the data type, 
        number of missing values and percentage of missing values for each column.
        
        Parameters:
        data (pandas.DataFrame): The dataframe to be analyzed for missing values
        
        Returns:
        pandas.DataFrame: A dataframe containing the data type, number of missing values and percentage of 
        missing values for each column in the input dataframe.
        """
        
        type_ = data.dtypes
        sum_na = data.isna().sum()
        per_na = np.round(data.isna().mean()*100,2).astype(str) + ' %'
        
        metrics = pd.DataFrame({'Type':type_, 'number na': sum_na, 'percent na':per_na})
        
        return metrics

    def statistics_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes basic statistics for numerical columns in a pandas DataFrame.

        Parameters:
        -----------
        data : pandas DataFrame
            The DataFrame for which to compute the statistics.

        Returns:
        --------
        pandas DataFrame
            A DataFrame containing the following statistics for each numerical column:
            - Non-Null: number of non-null values
            - range: range of values (max - min)
            - min: minimum value
            - quant25: 25th percentile
            - median: 50th percentile
            - quant75: 75th percentile
            - max: maximum value
            - mean: mean value
            - std: standard deviation
            - skew: skewness
            - kurtosis: kurtosis
        """
        num_data = data.select_dtypes(include=['int', 'float'])

        # Central Tendency
        mean = num_data.mean()
        median = num_data.median()

        # Quantiles
        q25 = num_data.quantile(0.25)
        q75 = num_data.quantile(0.75)

        # Dispersion
        rng = num_data.max() - num_data.min()
        std = num_data.std()

        skew = num_data.skew()
        kurtosis = num_data.kurtosis()

        metrics = pd.DataFrame({
            'Non-Null': num_data.count(),
            'range': rng,
            'min': num_data.min(),
            'quant25': q25,
            'median': median,
            'quant75': q75,
            'max': num_data.max(),
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurtosis': kurtosis
        })

        return metrics

    def plot_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Plot countplots for all categorical features in the given dataframe.

        Parameters:
        -----------
        data : pandas DataFrame
            The data to be analyzed.

        Returns:
        --------
        None
        """
        cat_attributes = data.select_dtypes(exclude=['int', 'float', 'datetime64[ns]'])

        for col in cat_attributes.columns:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=cat_attributes, ax=ax)
            ax.set_title("CountPlot " + col, fontsize=10)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:.1f}%'.format(height / len(cat_attributes) * 100),
                        ha="center")

