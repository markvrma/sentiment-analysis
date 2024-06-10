import pandas as pd
from sklearn.model_selection import train_test_split

# in progress
category_columns =  ['garage service', 'ease of booking', 'value for money', 'location',
       'length of fitting', 'change of date', 'tyre quality', 'wait time',
       'delivery punctuality', 'mobile fitter', 'advisor/agent service',
       'advisoragent service', 'extra charges', 'damage', 'balancing',
       'facilities', 'change of time', 'booking confusion', 'late notice',
       'discounts', 'refund not actioned', 'refund timescale',
       "mobile fitter didn't arrive", 'discount not applied',
       'tyre agedot code', 'failed payment', 'incorrect tyres sent',
       'call wait time', 'refund', 'no stock', 'response time',
       'tyre age/dot code']

def preprocess_category_data(file_path):
    df = pd.read_csv('./category.csv')
    X = df['sentence'].tolist()
    y = df[category_columns].values.tolist()
    # splitting
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    # Preprocessing steps...
    return X_train, X_eval, y_train, y_eval

def preprocess_sentiment_data(file_path):
    df = pd.read_csv('./sentiment.csv')
    
    #  combine 'sentence' and 'category' columns
    df['input'] = df['sentence'] + " [CATEGORY] " + df['category']

    # map sentiments to numerical values
    sentiment_mapping = {"positive": 1, "negative": 0}
    df['label'] = df['sentiment'].map(sentiment_mapping)

    # separate features and labels
    X = df['input'].tolist()
    y = df['label'].tolist()
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_eval, y_train, y_eval