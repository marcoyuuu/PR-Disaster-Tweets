import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('PR_Earthquake_Tweets_Jan2020.csv')

# Step 2: Filter rows where the 'Language' column is 'en' or 'es'
filtered_df = df[df['Language'].isin(['en', 'es'])]

# Step 3: Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_file.csv', index=False)

filtered_df = df[df['Language'].isin(['en', 'es'])]
print(f"Filtered row count: {len(filtered_df)}")
