from cgitb import text
from turtle import pos
from typing_extensions import Self
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from logger import Logger
import sys
from sklearn.preprocessing import LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords
import re

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

class DataCleaner:
    def __init__(self) -> None:
        """Initilize class."""
        try:
            
            nltk.download('wordnet')
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            
            self.logger = Logger("data_cleaner.log").get_app_logger()
            self.logger.info(
                'Successfully initialized data cleaner Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate data cleaner Object')
            sys.exit(1)

    def drop_duplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """
        df.drop_duplicates(inplace=True)
        self.logger.info(f'dropped duplicate values')

        return df
    def convert_to_datetime(self, df: pd.DataFrame,columns :list) -> pd.DataFrame:
        """
        convert column to datetime
        """

        df[columns] = df[columns].apply(pd.to_datetime)
        self.logger.info(f'converted {columns} to date time')

        return df

    def convert_to_string(self, df: pd.DataFrame, columns :list) -> pd.DataFrame:
        """
        convert columns to string
        """
        df[columns] = df[columns].astype(str)
        self.logger.info(f'converted {columns} to string')

        return df
    
    def convert_to_int(self, df: pd.DataFrame, columns :list) -> pd.DataFrame:
        """
        convert columns to string
        """
        df[columns] = df[columns].astype(int)
        self.logger.info(f'converted {columns} to int')

        return df
    
    def convert_to_object(self, df: pd.DataFrame, columns :list) -> pd.DataFrame:
        """
        convert columns to string
        """
        df[columns] = df[columns].astype(object)
        self.logger.info(f'converted {columns} to object')

        return df
    
    def convert_to_float(self, df: pd.DataFrame, columns :list) -> pd.DataFrame:
        """
        convert columns to string
        """
        df[columns] = df[columns].astype(float)
        self.logger.info(f'converted {columns} to float')

        return df

    def remove_whitespace_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        remove whitespace from columns
        """
        df.columns = [column.replace(' ', '_').lower() for column in df.columns]
        self.logger.info(f'removed white spaces from column name')

        return df

    def percent_missing(self, df: pd.DataFrame) -> float:
        """
        calculate the percentage of missing values from dataframe
        """
        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMising = missingCount.sum()
        self.logger.info(f'calculated missing value percentage')

        return round(totalMising / totalCells * 100, 2)

    def get_numerical_columns(self, df: pd.DataFrame) -> list:
        """
        get numerical columns
        """
        return df.select_dtypes(include=['number']).columns.to_list()

    def get_categorical_columns(self, df: pd.DataFrame) -> list:
        """
        get categorical columns
        """
        return  df.select_dtypes(include=['object','datetime64[ns]']).columns.to_list()

    def percent_missing_column(self, df: pd.DataFrame, columns:list) -> pd.DataFrame:
        """
        calculate the percentage of missing values for the specified column
        """
        rows=[]
        for col in columns:
            try:
                col_len = len(df[col])
                missing_count = df[col].isnull().sum()
                # print(f"{col} has {round(missing_count / col_len * 100, 2)}% of its data missing")
                rows.append([col,col_len,missing_count,round(missing_count / col_len * 100, 2),df[col].dtype])
            except KeyError:
                rows.append([col,"Not found","Not found","Not found","Not found"])
        return pd.DataFrame(data=rows,columns=["Col Name","Total","Missing","%","Data Type"]).sort_values(by="%",ascending=False)

            

    
    def fill_missing_values_categorical(self, df: pd.DataFrame, method: str,columns:list=[]) -> pd.DataFrame:
        """
        fill missing values with specified method
        """

        if len(columns)==0:
            columns = df.select_dtypes(include=['object','datetime64[ns]']).columns


        if method == "ffill":

            for col in columns:
                df[col] = df[col].fillna(method='ffill')

            self.logger.info(f'fill missing values using ffill')

            return df

        elif method == "bfill":

            for col in columns:
                df[col] = df[col].fillna(method='bfill')
            
            self.logger.info(f'fill missing values using bfill')

            return df

        elif method == "mode":
            
            for col in columns:
                df[col] = df[col].fillna(df[col].mode()[0])
            self.logger.info(f'fill missing values using mode')

            return df
        else:
            print("Method unknown")
            self.logger.error(f'failed to fill missing values; method unkown')

            return df

    def fill_missing_values_numeric(self, df: pd.DataFrame, method: str,columns: list =None) -> pd.DataFrame:
        """
        fill missing values with specified method
        """
        if(columns==None):
            numeric_columns = self.get_numerical_columns(df)
        else:
            numeric_columns=columns

        if method == "mean":
            for col in numeric_columns:
                df[col].fillna(df[col].mean(), inplace=True)
            self.logger.info(f'fill missing values by mean')


        elif method == "median":
            for col in numeric_columns:
                df[col].fillna(df[col].median(), inplace=True)
            self.logger.info(f'fill missing values by median')

        else:
            print("Method unknown")
            self.logger.error(f'failed to fill missing values; method unkown')
        
        return df

    def remove_nan_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        remove columns with nan values for categorical columns
        """

        categorical_columns = self.get_categorical_columns(df)
        for col in categorical_columns:
            df = df[df[col] != 'nan']
        self.logger.error(f'remove nan categorical values')

        return df

    def normalizer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        normalize numerical columns
        """
        norm = Normalizer()
        return pd.DataFrame(norm.fit_transform(df[self.get_numerical_columns(df)]), columns=self.get_numerical_columns(df))

    def min_max_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """

        minmax_scaler = MinMaxScaler()
        columns=["CompetitionDistance","Sales","Customers"]

        
        df[columns]=minmax_scaler.fit_transform(df[columns])
            
        return df

    def standard_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """
        try:
            standard_scaler = StandardScaler()
            columns=self.get_numerical_columns(df)
            columns.remove("Store")
        except:
            pass
        # columns=["CompetitionDistance","Sales","Customers"]

        
        df[columns]=standard_scaler.fit_transform(df[columns])
            # try:
            # except:
            #     print(f"error with {col}")
            #     pass

        return df
        
    

    def fill_mode(self, df, columns):
        """Fill missing data with mode."""
        for col in columns:
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except Exception:
                print(f'Failed to Fill {col} Data')
        return df
    
    def fill_zeros(self, df, columns):
        """Fill missing data with zeros."""
        for col in columns:
            try:
                df[col] = df[col].fillna(0)
            except Exception:
                print(f'Failed to Fill {col} Data')
        return df
    
    def feature_encodder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode features using LabelEncoder.
        """
        features = self.get_categorical_columns(df)
        try:
            features.remove("Date")
        except:
            pass
        for feature in features:
            encodder = LabelEncoder()
            encodder.fit(df[feature])
            df[feature] = encodder.transform(df[feature])
        return df

    def clean_links(self,df:pd.DataFrame,columns:list=[])->pd.DataFrame:
        if len(columns) is 0:
            columns=self.get_categorical_columns(df)
        for col in columns:
            try:
                df[col]=df[col].apply(lambda x:re.sub(r"\S*https?:\S*", "",x))
                df[col]=df[col].apply(lambda x:re.sub(r"\S*www.\S*", "", x))
            except:
                pass
            
        return df

    def clean_symbols(self,df:pd.DataFrame,columns:list=[])->pd.DataFrame:

        if len(columns) is 0:
            columns=self.get_categorical_columns(df)

        for col in columns:
            df[col]=df[col].apply(lambda x:re.sub(r'[^\w]', ' ',x))
        return df

    def clean_stopwords(self,df:pd.DataFrame,columns:list = []):
         
        if len(columns) is 0:
            columns=self.get_categorical_columns(df)

        for col in columns:
            df[col]=df[col].apply(lambda x:remove_stopwords(x))
        return df

    def convert_to_lower_case(self,df:pd.DataFrame,columns:list = []):
        columns=self.get_categorical_columns(df)
        for col in columns:
            df[col]=df[col].apply(lambda x: str(x).lower())
        return df

    def stem_word(self,df:pd.DataFrame,columns:list=[]):
        
        stemmer = PorterStemmer()
        
        def stem(txt:str):
            stemmed_words = []
            try:
                word_list = nltk.word_tokenize(txt)
            except:
                word_list=txt.split()
            
            for word in word_list:
                try:
                    stemmed_words.append(stemmer.stem(word)) 
                except:
                    stemmed_words.append(word)
            return ' '.join(stemmed_words)

        if len(columns) is 0:
            columns = self.get_categorical_columns(df)

        for col in columns:
            df[col] = df[col].apply(lambda x:stem(x))
        return df

        

    def lemantize(self,df:pd.DataFrame,columns:list=[]):

        wordnet_lemmatizer = WordNetLemmatizer()

        def leman(txt:str):
            leman_words = []
            try:
                word_list = nltk.word_tokenize(txt)
            except:
                word_list=txt.split()
            
            for word in word_list:
                try:
                    leman_words.append(wordnet_lemmatizer.lemmatize(word,pos='v'))
                except:
                    pass
            return ' '.join(leman_words)

        if len(columns) is 0:
            columns = self.get_categorical_columns(df)

        for col in columns:
            df[col] = df[col].apply(lambda x:leman(x))
        return df
    
    def trail_space_remove(self,df:pd.DataFrame,columns:list=[]):

        if len(columns) is 0:
            columns = self.get_categorical_columns(df)

        for col in columns:
            df[col] = df[col].apply(lambda x:re.sub(r' +', ' ',x))
            df[col] = df[col].apply(lambda x:x.strip())
        return df
    
    def drop_duplicated_words(self,df:pd.DataFrame,columns:list=[]):
        if len(columns) is 0:
            columns = self.get_categorical_columns(df)

        for col in columns:
            df[col] = df[col].apply(lambda x:", ".join(set((x).split())))
        return df
        
