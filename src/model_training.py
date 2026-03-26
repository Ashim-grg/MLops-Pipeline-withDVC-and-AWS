import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)-> pd.DataFrame:
    'Load data from a csv file'
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error occured while loading the data: %s',e)
        raise
    
def train_model(X_train:np.ndarray, y_train:np.ndarray, params:dict)->RandomForestClassifier:
    '''
    Train the RandomForest model
    :param X_train -> training features
    :param y_train -> training labels
    :param param -> Dictionary of hyper parameters
    :return Random forest classifier
    '''
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The number of samples in X_train and y_train must be same.')
        
        logger.debug('Initializing RandomForest model with parameters %s',params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples',X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug('Model Training Completed.')
        
        return clf
    except ValueError as e:
        logger.error('Value Error during model training: %s',e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s',e)
        raise
    
def save_model(model,file_path:str)-> None:
    '''
    Saved the trained model to a file.
    :param model: Trained model object
    :param file_path: Path to save the model
    '''
    
    try:
        # Ensure the directory exist
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug('Model Saved to: %s',file_path)
    except FileExistsError as e:
        logger.error('File path not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error occured while saving the model: %s',e)
        raise
    
def main():
    try:
        params = {'n_estimators':25,'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        clf = train_model(X_train,y_train,params)
        
        model_save_pth = 'models/model.pkl'
        save_model(clf,model_save_pth)
    except Exception as e:
        logger.error('Failed to complete the model training process: %s',e)
        print(f'Error: ',e)
        
if __name__ == '__main__':
    main()
        
    
    
    