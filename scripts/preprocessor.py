import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import warnings
from logger import Logger
import json


from data_describe.text.text_preprocessing import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from data_cleaner import DataCleaner

warnings.filterwarnings('ignore')

sys.path.append('../')


class Processor:
    def __init__(self) -> None:
        """Initilize class."""
        try:
            self.logger = Logger("preprocessor.log").get_app_logger()
            self.logger.info('Successfully initialized processor Object')
            self.cleaner=DataCleaner()
        except Exception:
            self.logger.exception('Failed to processor util Object')
            sys.exit(1)
    
    def prepare_text(self,df:pd.DataFrame,columns:list=[]):
        self.logger.info("Preparing texts")
        if len(columns) is 0:
            columns = self.cleaner.get_categorical_columns(df)
        targeted_df=df[columns]
        pipeline= Pipeline(steps=[
            ('link_cleanner',FunctionTransformer(self.cleaner.clean_links, kw_args={"columns":columns},validate=False)),
            ('symbol_cleanner',FunctionTransformer(self.cleaner.clean_symbols, kw_args={"columns":columns},validate=False)),
            ('lower_casing',FunctionTransformer(self.cleaner.convert_to_lower_case, kw_args={"columns":columns},validate=False)),
            ('remove_stop_word',FunctionTransformer(self.cleaner.clean_stopwords, kw_args={"columns":columns},validate=False)),
            ('stemmer',FunctionTransformer(self.cleaner.stem_word, kw_args={"columns":columns},validate=False)),
            ('lemmatization',FunctionTransformer(self.cleaner.lemantize, kw_args={"columns":columns},validate=False)),
            ('trail_space_remover',FunctionTransformer(self.cleaner.trail_space_remove, kw_args={"columns":columns},validate=False)),
            ('remove_duplicate words',FunctionTransformer(self.cleaner.drop_duplicated_words, kw_args={"columns":columns},validate=False)),
        ])

        transformed=pipeline.fit_transform(targeted_df)
        
        df[columns]=transformed
        return df
    
    def prepare_tuner(self,train_df:pd.DataFrame):


        prompt=""
        for ind in train_df.index:
            prompt += "Task: Generate Analyst Average Score\n\n"
            for col in train_df.columns:
                if train_df.loc[ind,col] is not '':
                    prompt += f"{col}: {train_df.loc[ind,col]}\n\n"
            prompt += "-- --\n\n"
        try:
            with open('../data/tuner.txt', 'w', encoding="utf-8") as f:
                f.write(prompt)
            print("tuner prepared successfuly")
        except:
            print("Failed to prepare tuner")

    def prepare_job_description_text(self,df:pd.DataFrame):
        
        for ind in df.index:
    
            tokens_json=json.dumps(df.loc[ind,'tokens'])
            tokens=pd.read_json(tokens_json)
            try:
                entities=tokens.groupby(["entityLabel"])
                tokens['text'] = tokens[['entityLabel','text']].groupby(['entityLabel'])['text'].transform(lambda x: ', '.join(x))
                entities=tokens[['entityLabel','text']].drop_duplicates()
                entities['text'] = entities['entityLabel'] + ": " + entities['text']
                entities.drop('entityLabel',axis=1,inplace=True)
            except:
                pass    
            tokens=""

            for idx in entities.index:
                tokens += (f"{entities.loc[idx,'text']};")


            relation_json=json.dumps(df.loc[ind,'relations'])
            relation=pd.read_json(relation_json)

            try:
                relationLabels=relation['relationLabel'].unique()
            except:
                relationLabels=[]

            relationLabels=", ".join(relationLabels)

            df['tokens'][ind]=tokens
            df['relations'][ind]=relationLabels

        return df

    def prepare_job_description_tuner(self,train_df:pd.DataFrame):


        prompt=""
        for ind in train_df.index:
            prompt += "Task: Extract job description entites from this text\n\n"
            

            prompt += f"Description: {train_df.loc[ind,'document']}\n\n"
            tokens = "\n\n".join(train_df.loc[ind,'tokens'].split(';'))
            prompt += f"{tokens}\n\n"
            prompt += "-- --\n\n"
        try:
            with open('../data/job-tuner.txt', 'w', encoding="utf-8") as f:
                f.write(prompt)
            print("tuner prepared successfuly")
        except:
            print("Failed to prepare tuner")
        

            