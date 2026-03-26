import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    'Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming'
    
    ps = PorterStemmer()
    
    if not isinstance(text,str):
        text = str(text)
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove all non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words 
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    'Preprocess the dataframe by encoding the target column, remove duplicates, transform text column'
    try:
        logger.debug('Starting the Preprocessing for DataFrame')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column Encoded')
        
        # Removing duplicates
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates Removed.')
        
        # Apply text transformation to specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text Column Transformed.')
        return df
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s',e)
        raise

def main(test_column='text',target_column='target'):
    'Main Function to load raw data, preprocess it, and save the processed data.'
    
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded Properly')
        
        # Transform the data
        train_processed_data = preprocess_df(train_data,test_column,target_column)
        test_processed_data = preprocess_df(test_data,test_column,target_column)
        
        # Store the data inside data/processed
        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
        
        logger.debug('Processed data saved to %s',data_path)
    except FileNotFoundError as e:
        logger.error('File not Found: %s',e)
    except pd.errors.EmptyDataError as e:
        logger.error('No Data: %s',e)
    except Exception as e:
        logger.error('Failed to Complete the Data Transformation process: %s',e)
        print(f'Error: ',e)
    
if __name__ == '__main__':
    main()