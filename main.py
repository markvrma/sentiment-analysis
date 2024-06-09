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

finaldf = pd.DataFrame(columns=['sentence','category','sentiment'])

# now i need to take each value in the columns, split it into seperate columns or add it to a new df
# iterate through each column
columns_to_iterate = ['sentiment1', 'sentiment2', 'sentiment3', 'sentiment4',
       'sentiment5', 'sentiment6', 'sentiment7', 'sentiment8', 'sentiment9',
       'sentiment10', 'sentiment11', 'sentiment12', 'sentiment13',
       'sentiment14']

for row in tqdm(org_clean.itertuples(index=True), total=len(org_clean)): # apply Poolexecutor
    for col in columns_to_iterate:
        value = getattr(row, col)

        # skip nan values
        if pd.notna(value):
            category,sentiment = split_at_last_word(value)
            
        
        if sentiment not in ['positive', 'negative']:
            continue

        # apply this to another dataframe
        if pd.notna(value):
            new_row_data = {'sentence': row.Sentence, 'category': category, 'sentiment': sentiment}
            finaldf = finaldf._append(new_row_data, ignore_index=True)
        # finaldf.append({'sentence':row['Sentence'], 'category':category, 'sentiment':sentiment}, ignore_index=True)

print(finaldf)
unique_values_per_column = {col: finaldf[col].unique() for col in finaldf.columns[1:]}
no_unique_values_per_column = {col: len(finaldf[col].unique()) for col in finaldf.columns[1:]}

print(unique_values_per_column)
print(no_unique_values_per_column)
finaldf.to_csv('final.csv')