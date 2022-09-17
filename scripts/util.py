# from copyreg import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api as dvc
from logger import Logger
from sklearn.model_selection import train_test_split
import io 
import sys
import pickle
sys.path.append('../')

class Util:
    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            self.logger = Logger("utility.log").get_app_logger()
            self.logger.info(
                'Successfully initialized util Object')
        except Exception:
            self.logger.exception(
                'Failed to initialized util Object')
            sys.exit(1)
    def read_from_file(self,path,low_memory=True):
        """
            Load data from a csv file
        """
        try:
            df = pd.read_csv(path)
            self.logger.info(f"successsfuly read {path}")
            return df
        except FileNotFoundError:
            self.logger.error(f"failed to read {path}; file not found")

            print("File not found.")
    def read_from_dvc(self,path,repo,rev,low_memory=True):
        
        """
            Load data from a dvc storage
        """
        try:
            data = dvc.read(path=path,repo=repo, rev=rev,encoding="utf8")
            df = pd.read_csv(io.StringIO(data),low_memory=low_memory)
            self.logger.info(f"successsfuly read {path} from dvc")

            return df
        except Exception as e:
            self.logger.error(f"failed to read {path}; {e}")

            print("Something went wrong!",e)
    
    def read_model_dvc(self,path,repo,rev,low_memory=True):
        
        """
            Load data from a dvc storage
        """
        model = None
        try:
            data = dvc.read(path=path,repo=repo, rev=rev)
            model = pickle.load(io.StringIO(data))
        except Exception as e:
            print(f"{e}")
        return model


    def train_test_split(self, input_data:tuple, size:tuple)-> list:
        """
        Split the data into train, test and validation.
        """
        
        X,Y = input_data
        train_size,test_size=size

        train_x=X.iloc[:round(train_size * X.shape[0])]
        test_x=X.iloc[-round(test_size * X.shape[0]):]
        
        train_y=Y.iloc[:round(train_size * len(Y))]
        test_y=Y.iloc[-round(test_size * len(Y)):]
        
        # train_x, temp_x, train_y, temp_y = train_test_split(X, Y, train_size=size[0], test_size=size[1]+size[2], random_state=42)
        # test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y, train_size=size[1]/(size[1]+size[2]), test_size=size[2]/(size[1]+size[2]), random_state=42)
        return [train_x, train_y, test_x, test_y]