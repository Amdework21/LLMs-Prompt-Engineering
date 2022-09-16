
import pandas as pd
from builtins  import Str1
from string  import Str
import numpy as np
from log_help import App_Logger

app_logger = App_Logger("../logs/data_cleaner.log").get_app_logger()


class DataCleaner:
    def __init__(self, df: pd.DataFrame, deep=False) -> None:
        """
        Returns a DataCleaner Object with the passed DataFrame Data set as its own DataFrame
        Parameters
        ----------
        df:
            Type: pd.DataFrame

        Returns
        -------
        None
        """
        self.logger = App_Logger(
            "../logs/data_cleaner.log").get_app_logger()
        if(deep):
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns in the list are removed
        Parameters
        ----------
        columns:
            Type: list

        Returns
        -------
        pd.DataFrame
        """
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def separate_date_time_column(self, column: str, col_prefix_name: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns is split to date and time new columns adding a prefix string to both
        Parameters
        ----------
        column:
            Type: str
        col_prefix_name:
            Type: str

        Returns
        -------
        pd.DataFrame
        """
        try:

            self.df[f'{col_prefix_name}Date'] = pd.to_datetime(
                self.df[column]).dt.date
            self.df[f'{col_prefix_name}Time'] = pd.to_datetime(
                self.df[column]).dt.time

            return self.df

        except:
            print("Failed to separate the date-time column")

    def separate_date_column(self, date_column: str, drop_date=True) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(date_column)
            self.df.insert(date_index + 1, 'Year', self.df[date_column].apply(
                lambda x: x.date().year))
            self.df.insert(date_index + 2, 'Month', self.df[date_column].apply(
                lambda x: x.date().month))
            self.df.insert(date_index + 3, 'Day',
                           self.df[date_column].apply(lambda x: x.date().day))

            if(drop_date):
                self.df = self.df.drop(date_column, axis=1)
        except:
            print("Failed to separate the date to its components")

    def change_column_to_date_type(self, col_name: str) -> None:
        try:
            self.df[col_name] = pd.to_datetime(self.df[col_name])
        except:
            print('failed to change column to Date Type')
        self.logger.info(
            f"Successfully changed column {col_name} to Date Type")

    def remove_nulls(self) -> pd.DataFrame:
        return self.df.dropna()

    def add_season_col(self, month_col: str) -> None:
        # helper function
        def get_season(month: int):
            if(month <= 2 or month == 12):
                return 'Winter'
            elif(month > 2 and month <= 5):
                return 'Spring'
            elif(month > 5 and month <= 8):
                return 'Summer'
            else:
                return 'Autumn'

        try:
            month_index = self.df.columns.get_loc(month_col)
            self.df.insert(month_index + 1, 'Season',
                           self.df[month_col].apply(get_season))

        except:
            print("Failed to add season column")
        self.logger.info(f"Successfully added season column to {month_col}")

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns data types are changed to the specified data type
        Parameters
        ----------
        cols:
            Type: list
        data_type:
            Type: str

        Returns
        -------
        pd.DataFrame
        """
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Failed to change columns type')
        self.logger.info(f"Successfully changed columns type to {data_type}")
        return self.df

    def remove_single_value_columns(self, unique_value_counts: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame where columns with a single value are removed
        Parameters
        ----------
        unique_value_counts:
            Type: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        drop_cols = list(
            unique_value_counts.loc[unique_value_counts['Unique Value Count'] == 1].index)
        return self.df.drop(drop_cols, axis=1, inplace=True)

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Returns a DataFrame where duplicate rows are removed
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        removables = self.df[self.df.duplicated()].index
        return self.df.drop(index=removables, inplace=True)

    def fill_numeric_values(self, missing_cols: list, acceptable_skewness: float = 5.0) -> pd.DataFrame:
        """
        Returns a DataFrame where numeric columns are filled with either median or mean based on their skewness
        Parameters
        ----------
        missing_cols:
            Type: list
        acceptable_skewness:
            Type: float
            Default value = 5.0

        Returns
        -------
        pd.DataFrame
        """
        df_skew_data = self.df[missing_cols]
        df_skew = df_skew_data.skew(axis=0, skipna=True)
        for i in df_skew.index:
            if(df_skew[i] < acceptable_skewness and df_skew[i] > (acceptable_skewness * -1)):
                value = self.df[i].mean()
                self.df[i].fillna(value, inplace=True)
            else:
                value = self.df[i].median()
                self.df[i].fillna(value, inplace=True)

        return self.df

    def add_columns_from_another_df_using_column(self, from_df: pd.DataFrame, base_col: str, add_columns: list) -> pd.DataFrame:
        try:
            new_df = self.df.copy(deep=True)
            from_df.sort_values(base_col, ascending=True, inplace=True)
            for col in add_columns:
                col_index = from_df.columns.tolist().index(col)
                new_df[col] = new_df[base_col].apply(
                    lambda x: from_df.iloc[x-1, col_index])

            return new_df

        except:
            print('Failed to add columns from other dataframe')

    def fill_non_numeric_values(self, missing_cols: list, ffill: bool = True, bfill: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame where non-numeric columns are filled with forward or backward fill
        Parameters
        ----------
        missing_cols:
            Type: list
        ffill:
            Type: bool
            Default value = True
        bfill:
            Type: bool
            Default value = False

        Returns
        -------
        pd.DataFrame
        """
        for col in missing_cols:
            if(ffill == True and bfill == True):
                self.df[col].fillna(method='ffill', inplace=True)
                self.df[col].fillna(method='bfill', inplace=True)

            elif(ffill == True and bfill == False):
                self.df[col].fillna(method='ffill', inplace=True)

            elif(ffill == False and bfill == True):
                self.df[col].fillna(method='bfill', inplace=True)

            else:
                self.df[col].fillna(method='bfill', inplace=True)
                self.df[col].fillna(method='ffill', inplace=True)

        return self.df

    def create_new_columns_from(self, new_col_name: str, col1: str, col2: str, func) -> pd.DataFrame:
        """
        Returns a DataFrame where a new column is created using a function on two specified columns
        Parameters
        ----------
        new_col_name:
            Type: str
        col1:
            Type: str
        col2:
            Type: str
        func:
            Type: function

        Returns
        -------
        pd.DataFrame
        """
        try:
            self.df[new_col_name] = func(self.df[col1], self.df[col2])
        except:
            print("failed to create new column with the specified function")

        return self.df

    def convert_bytes_to_megabytes(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where columns value is changed from bytes to megabytes

        Args:
        -----
        columns: 
            Type: list

        Returns:
        --------
        pd.DataFrame
        """
        try:
            megabyte = 1*10e+5
            for col in columns:
                self.df[col] = self.df[col] / megabyte
                self.df.rename(
                    columns={col: f'{col[:-7]}(MegaBytes)'}, inplace=True)

        except:
            print('failed to change values to megabytes')

        return self.df

    def fix_outlier_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where outlier of the specified columns is fixed
        Parameters
        ----------
        columns:
            Type: list

        Returns
        -------
        pd.DataFrame
        """
        try:
            for column in columns:
                self.df[column] = np.where(self.df[column] > self.df[column].quantile(
                    0.95), self.df[column].median(), self.df[column])
        except:
            print("Cant fix outliers for each column")

    def replace_outlier_with_median(self, dataFrame: pd.DataFrame, feature: Str) -> pd.DataFrame:

        Q1 = dataFrame[feature].quantile(0.25)
        Q3 = dataFrame[feature].quantile(0.75)
        median = dataFrame[feature].quantile(0.50)

        IQR = Q3 - Q1

        upper_whisker = Q3 + (1.5 * IQR)
        lower_whisker = Q1 - (1.5 * IQR)

        dataFrame[feature] = np.where(
            dataFrame[feature] > upper_whisker, median, dataFrame[feature])
        dataFrame[feature] = np.where(
            dataFrame[feature] < lower_whisker, median, dataFrame[feature])
        self.logger.info(f"Outlier for {feature} is fixed")

        return dataFrame

    def standardized_column(self, columns: list, new_name: list, func) -> pd.DataFrame:
        """
        Returns a DataFrame where specified columns are standardized based on a given function and given new names after
        Parameters
        ----------
        columns:
            Type: list
        new_name:
            Type: list
        func:
            Type: function

        Returns
        -------
        pd.DataFrame
        """
        try:
            assert(len(columns) == len(new_name))
            for index, col in enumerate(columns):
                self.df[col] = func(self.df[col])
                self.df.rename(columns={col: new_name[index]}, inplace=True)
            self.logger.info(f"Columns are standardized")
        except AssertionError:
            print('size of columns and names provided is not equal')

        except:
            print('standardization failed')
        return self.df

    def optimize_df(self) -> pd.DataFrame:
        """
        Returns the DataFrames information after all column data types are optimized (to a lower data type)
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        data_types = self.df.dtypes
        optimizable = ['float64', 'int64']
        try:
            for col in data_types.index:
                if(data_types[col] in optimizable):
                    if(data_types[col] == 'float64'):
                        # downcasting a float column
                        self.df[col] = pd.to_numeric(
                            self.df[col], downcast='float')
                    elif(data_types[col] == 'int64'):
                        # downcasting an integer column
                        self.df[col] = pd.to_numeric(
                            self.df[col], downcast='unsigned')
            self.logger.info(f"DataFrame optimized")
            return self.df

        except:
            print('Failed to optimize')

    def save_clean_data(self, name: str):
        """
        The objects dataframe gets saved with the specified name 
        Parameters
        ----------
        name:
            Type: str

        Returns
        -------
        None
        """
        try:
            self.df.to_csv(name, index=False)
            self.logger.info(f"DataFrame saved")
        except:
            print("Failed to save data")
