import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import yaml
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path:str)->dict:
    'Load Parameters from a yaml file'
    try:
        with open(param_path,'r') as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters retrieved from %s',param_path)
        return params
    except FileNotFoundError as e:
        logger.error('File not found. %s',e)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml Error. %s',e)
    except Exception as e:
        logger.error('Unexpected Error %s',e)
        raise

def load_model(file_path:str):
    'Load trained model from a file.'
    try:
        with open(file_path,'rb') as f:
            model = pickle.load(f)
        logger.debug('Model Loaded from %s',file_path)
        return model
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error occured while loading the model: %s',e)
        raise
    
def load_data(file_path:str)-> pd.DataFrame:
    'Load Data from csv file.'
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error occured while loading the data: %s',e)
        raise
    
def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    'Evaluate Model return the evaluation metrics.'
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)
        
        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('Model Evaluation metrics calculated.')
        return metrics_dict
    except Exception as e:
        logger.error('Error During model evaluation %s',e)
        raise
    
def save_metrics(metrics:dict,file_path:str)-> None:
    'Save the evaluation metrics to a json file'
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Metrics saved to %s',file_path)
    except Exception as e:
        logger.debug('Error occured while saving the metrics: %s',e)
        raise
    
def main():
    try:
        params = load_params(param_path='params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        
        metrics = evaluate_model(clf,X_test,y_test)
        save_metrics(metrics,'reports/metrics.json')
        
        with Live(save_dvc_exp = True) as live:
            live.log_metric('accuracy',accuracy_score(y_test,y_test))
            live.log_metric('precision',precision_score(y_test,y_test))
            live.log_metric('recall',recall_score(y_test,y_test))

            live.log_params(params)
    except Exception as e:
        logger.error('Falied to complete the model evaluation process: %s',e)
        print(f'Error: ',e)

if __name__ == '__main__':
    main()