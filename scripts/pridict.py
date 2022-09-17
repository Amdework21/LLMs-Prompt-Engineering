import warnings
from logger import Logger
import cohere
import pandas as pd
from IPython.display import display
from config import api_key
import sys
warnings.simplefilter('ignore')

sys.path.append('../')

class Predict:
    def __init__(self):
        """Initilize class."""
        try:
            self.logger = Logger("predictor.log").get_app_logger()
            self.api_key=api_key
            self.logger.info('Successfully initialized predictor Object')
        except Exception:
            self.logger.exception('Failed to processor util Object')
            sys.exit(1)
        
    def predict(self,train_df:pd.DataFrame,test:pd.DataFrame, model="xlarge"):

        test_df=test.drop(test.columns[len(test.columns)-1],axis=1)

        prompt=""
        
        for ind in train_df.index:
            prompt += "Task: Generate Analyst Average Score\n\n"
            for col in train_df.columns:
                prompt += f"{col}: {train_df.loc[ind,col]}\n\n"
            prompt += "-- --\n\n"
        
        prompt += "Task: Generate Analyst Average Score\n\n"
        for col in test_df.columns:
            ind=test_df.index[0]
            prompt += f"{col}: {test_df.loc[ind,col]}\n\n"

        prompt += f"{test.columns[len(test.columns)-1]}: "
        # prompt = "Task: Score News relevance\n\n"+prompt
        
        co = cohere.Client(f'{self.api_key}')
        response = co.generate(
        model=model,
        prompt=prompt.strip(),
        max_tokens=4,
        temperature=0.9,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["-- --"],
        return_likelihoods='NONE')
        print('Prediction: {}'.format(response.generations[0].text))

        return prompt

    # def extract_entities(self,train_df:pd.DataFrame,test:pd.DataFrame, model="xlarge"):

    #         test_df=test.drop(test.columns[len(test.columns)-1],axis=1)

    #         prompt=""
            
    #         for ind in train_df.index:
    #             prompt += "Task: Extract job description entites from this text\n\n"
    #             for col in train_df.columns:
    #                 prompt += f"{col}: {train_df.loc[ind,col]}\n\n"
    #             prompt += "-- --\n\n"
            
    #         prompt += "Task: Extract job description entites from this text\n\n"
    #         for col in test_df.columns:
    #             ind=test_df.index[0]
    #             prompt += f"{col}: {test_df.loc[ind,col]}\n\n"

    #         prompt += f"{test.columns[len(test.columns)-1]}: "
            
    #         # co = cohere.Client(f'{self.api_key}')
    #         # response = co.generate(
    #         # model=model,
    #         # prompt=prompt.strip(),
    #         # max_tokens=4,
    #         # temperature=0.9,
    #         # k=0,
    #         # p=0.75,
    #         # frequency_penalty=0,
    #         # presence_penalty=0,
    #         # stop_sequences=["-- --"],
    #         # return_likelihoods='NONE')
    #         # print('Prediction: {}'.format(response.generations[0].text))

    #         return prompt

    def extract_entities(self,train_df:pd.DataFrame,test:pd.DataFrame, model="xlarge"):

        test_df=test.drop(test.columns[len(test.columns)-1],axis=1)

        prompt=""

        for ind in train_df.index:
            prompt += "Task: Extract job description entites from this text\n\n"
            

            prompt += f"Description: {train_df.loc[ind,'document']}\n\n"
            tokens = "\n\n".join(train_df.loc[ind,'tokens'].split(';'))
            prompt += f"{tokens}\n\n"
            prompt += "-- --\n\n"
        
        for ind in test_df.index:
            prompt += "Task: Extract job description entites from this text\n\n"
            

            prompt += f"Description: {test_df.loc[ind,'document']}\n\n"
            
        co = cohere.Client(f'{self.api_key}')
        response = co.generate(
        model=model,
        prompt=prompt.strip(),
        max_tokens=50,
        temperature=0.9,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["-- --"],
        return_likelihoods='NONE')
        print('Prediction: {}'.format(response.generations[0].text))
        
        return prompt
