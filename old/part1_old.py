from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Example categorical data
data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green']})

# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data = encoder.fit_transform(data[['Color']])

# Get new column names for the encoded features
columns = encoder.get_feature_names_out(['Color'])  # Use get_feature_names for older versions

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=columns)

print(encoded_df)