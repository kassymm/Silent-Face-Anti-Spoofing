import pandas as pd
import json

# Read the JSON file
with open('test_label.json', 'r') as file:
    data = json.load(file)

# Create a dataframe from the JSON data with orient='index'
df = pd.DataFrame.from_dict(data, orient='index')

# Print the first 10 rows
print(df.head(10))
