import pandas as pd
import numpy as np
from tqdm import tqdm


def split_at_last_word(string):
    # Strip any trailing whitespace
    string = string.rstrip()
    
    # Find the last space in the string
    last_space_index = string.rfind(' ')
    
    # If no space is found, return the original string as the last word
    if last_space_index == -1:
        return '', string
    
    # Split the string into two parts
    first_part = string[:last_space_index]
    last_word = string[last_space_index + 1:]
    
    return first_part, last_word



org = pd.read_csv('./Evaluation-dataset.csv')
org_clean = org.dropna(subset=['sentiment1'])
org_clean = org_clean.dropna(how='all',axis=1)
columns = ['sentence','garage service', 'ease of booking', 'value for money', 'location',
       'length of fitting', 'change of date', 'tyre quality', 'wait time',
       'delivery punctuality', 'mobile fitter', 'advisor/agent service',
       'advisoragent service', 'extra charges', 'damage', 'balancing',
       'facilities', 'change of time', 'booking confusion', 'late notice',
       'discounts', 'refund not actioned', 'refund timescale',
       "mobile fitter didn't arrive", 'discount not applied',
       'tyre agedot code', 'failed payment', 'incorrect tyres sent',
       'call wait time', 'refund', 'no stock', 'response time',
       'tyre age/dot code']

category_df = pd.DataFrame(columns=columns)
category_df[columns[1:]] = 0

sentiment_df = pd.DataFrame(columns=['sentence','category', 'sentiment'])

# now i need to take each value in the columns, split it into seperate columns or add it to a new df
# iterate through each column
columns_to_iterate = ['sentiment1', 'sentiment2', 'sentiment3', 'sentiment4',
       'sentiment5', 'sentiment6', 'sentiment7', 'sentiment8', 'sentiment9',
       'sentiment10', 'sentiment11', 'sentiment12', 'sentiment13',
       'sentiment14']

for row in tqdm(org_clean.itertuples(index=True), total=len(org_clean)): # apply Poolexecutor
    category_arr = []
    for col in columns_to_iterate:
        value = getattr(row, col)

        # skip nan values
        if pd.notna(value):
            category,sentiment = split_at_last_word(value)
            
        
        if sentiment not in ['positive', 'negative']:
            continue

        # depending on the category make that one and the others zeros
        if pd.notna(value):
            new_row_data = {'sentence': row.Sentence, 'category': category, 'sentiment': sentiment}
            category_arr.append(category)
            sentiment_df = sentiment_df._append(new_row_data, ignore_index=True)
    

    if len(category_arr) > 0:
        # new_row = pd.Series(0, index=category_df.columns)
        new_row = {col: 0 for col in columns}
        new_row['sentence'] = row.Sentence
        for category in category_arr:
            new_row[category] = 1
        
        category_df = category_df._append(new_row, ignore_index=True)
    

sentiment_df.to_csv('sentiment.csv')
category_df.to_csv('category.csv')