import pandas as pd
import numpy as np
from urllib.request import urlretrieve    # Library to access datasets from net via a url
import os

# For reading files location of file starting with a dot within single quotes is compulsory...
file = pd.read_csv('./Pandas Library/all_weekly_excess_deaths.csv')  # Reading of files is different for every type of file in pandas hence, more versatile than Numpy
url = 'https://gist.githubusercontent.com/aakashns/f6a004fa20c84fec53262f9a8bfee775/raw/f309558b1cf5103424cef58e2ecb8704dcd4d74c/italy-covid-daywise.csv'
# The url of the dataset
urlretrieve(url, 'italy-covid-daywise.csv')
# Type of the dataframe of the file
print(type(file))
file1 = pd.read_csv('italy-covid-daywise.csv')
print(file1)
# A dataframe in pandas is a 2d data structure that holds the necessary data in form of a table or a 2d matrix

print(file1.info())    # Getting information of the dataframe

print(file1.describe())    # Getting detailed information of the dataframe (the information is purely mathematical)
print(file1.columns)    # Getting the details of the columns in the dataframe
print(file1.shape)      # Getting the row x column number

# Accessing the Keys or the Values in pandas can be done by indexing... 
print(file1['new_cases'])               # Accessing entire column
print(file1.loc[243])                   # Accessing an entire row by its serial number
print((int)(file1['new_cases'][243]))    # Accessing a particular element of a specific column
print((int)(file1.at[243, 'new_cases']))
print(file1.loc[108:113])                 # Accessing a set of rows
print(file1[['new_cases', 'new_tests']])   # Accessing a set of columns
print("Head",file1.head(4))                    # Accessing the first few rows of the dataframe
print("Tail",file1.tail(5))                    # Accessing the last few rows of the dataframe
file1.fillna(0)                 # Filling the entries of null with 0...

'''The Columns in the dataframe are essentially a data structure called Series, a Numpy array having more properties and methods...'''

# We can create our subsets of data using the dataframe since the columns are Numpy Array
cases = file1[['date', 'new_cases']]     # A subset of a dataframe is a 2d array
print(cases)
file1.drop(index=2, columns='region_code')    # Removing column by its index and heading...
print(cases.shape)     # It should be 248 rows x 2 columns
# A subset of a dataframe is also a dataframe
cases1 = file1[['new_cases']]
cases1 = cases1.astype(int)      # Converting a dataframe float to integer, only works when the all the columns are of float data type... 
dates = file1[['date']]
new_df = [[dates, cases1]]    # The subsets of the dataframe can be clubbed together, and it will become a list and not a dataframe
df_array = np.array(new_df)    # This converts it to an array hence a dataframe... the dimensions will be the number of pre-existing columns +1
df = file1.copy()        # Method to copy a dataframe

print(file1['new_tests'].first_valid_index())     # Getting the first index in a column which is not nan
print(file1['new_cases'].last_valid_index())      # Getting the last index in a column which is nan
print(file1.sample(10))        # Getting a random sample of 10 rows from the dataset

total_cases = file1['new_cases'].sum()       # Getting the total number of the new cases
total_deaths = file1['new_deaths'].sum()     # Getting the total number of deaths
print((int)(total_cases), (int)(total_deaths))

file1['COD'].nunique()    # Count the number of unique values in the column...

# Querying Rows and Columns in a Dataframe
high = file1['new_cases'] > 1000
print(high)      # This dataframe stores a boolean value
low = file1['new_cases'] < 100
print(low)      # Conditional dataframes usually store boolean values and can be type casted to string
high_index = file1[high]
print(high_index)    # This dataframe stores the index of the high covid cases
file1['positive_rate'] = file1['new_cases']/file1['new_tests']    # Creating a new column
print(file1)
file1.drop(columns=['positive_rate'], inplace=True)
# Using the string property and slicing the first two characters
file1['Month'] = file1['Order Date'].str[0:2]
data = file1.dropna()     # Dropping null rows
print(file1)

# Sorting Rows and Columns
print(file1.sort_values('new_deaths', ascending=False))   # Sorting data in descending order
print(file1.sort_values('new_deaths', ascending=False).head(5))    # Sorting data im descending order and getting the top 5 values using head or bottom values using tail
# The lambda data type creates a user defined function and performs it on the entire column sequence
data['City'] = data['Purchase Address'].apply(lambda x: x.split(',')[1])
data.head(3)

# Since this data is ordered by date this is also termed as time series data (related to time)
file1['date'] = pd.to_datetime(file1['date'])   # Converting to date time data, now we can other date properties of the data
file1['year'] = pd.DatetimeIndex(file1['date']).year    # DatetimeIndex property is used
print(file1.columns)
file1['day'] = pd.DatetimeIndex(file1['date']).day
print(file1.columns)
file1['weekdays'] = pd.DatetimeIndex(file1['date']).weekday
print(file1.columns)
file1['month'] = pd.DatetimeIndex(file1['date']).month
# Printing file1 which has file1 month equal to 5
print(file1[file1['month'] == 5])    # Data of 5th Month (May)
may = file1[file1['month'] == 5]
# Must be placed in Square brackets while extracting
may_matrix = may[['new_cases', 'new_tests', 'new_deaths']]    # Extracting information from May dataframe
may_matrix = may_matrix.astype('int32')
print("Data from May :",may_matrix)
avg = file1['new_cases'].mean()       # Mean is a single value and not a set of values
avg1 = file1[file1['weekdays'] == 6]
avg1 = avg1['new_cases'].mean()
print(avg, avg1)

# Grouping and Aggregation of Data (must read a full documentation)
month_file = file1.groupby('month')[['new_cases', 'new_tests', 'new_deaths']].sum()          # Grouping data by month and evaluating the sum of each month
print(month_file)
month_file1 = file1.groupby('month')[['new_cases', 'new_tests', 'new_deaths']].mean()      # Grouping data by month and evaluating the mean of each month
print(month_file1)
file1['total_tests'] = file1['new_tests'].cumsum()       # It evaluates the cumulative sum or running sum
file1['total_deaths'] = file1['new_deaths'].cumsum()
file1['total_cases'] = file1['new_cases'].cumsum()    # Similarly we can do cumulative max, min and product
print(file1.shape)

# Merging data from multiple sources
urlretrieve('https://gist.githubusercontent.com/aakashns/8684589ef4f266116cdce023377fc9c8/raw/99ce3826b2a9d1e6d0bde7e9e559fc8b6e9ac88b/locations.csv','locations.csv')
location = pd.read_csv('locations.csv')
# Getting locations of Italy from the location dataframe
location[location['location'] == 'Italy']
file1['location'] = 'Italy'
merged = file1.merge(location, on='location')   # Merging location file at the column location in the file1
print(merged)
merged['cases_per_million'] = merged['total_cases'] * 1e6 / merged['population']
print(merged)

# Writing data back to files
result = merged
result.to_csv('results.csv', index=True)

# Basic Plotting with Pandas Library
r1 = result['new_cases'].plot()
r2 = result['total_deaths'].plot(kind='bar')

files = [file for file in os.listdir('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data')]
data = pd.DataFrame()     # Creating an empty dataframe 
for file in files:
    df = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/'+file)     # reading all files location
    data = pd.concat([data, df])    # Concatenating them into the basic DataFrame

data.replace(to_replace = np.nan, value=0)       # Replacing null values with 0
data.dropna()     # Removing entire row
data.dropna(axis=0)     # Removing an entire column
data.dropna(how='all')     # Removing row if all entries are null
data.corr()     # Provides a table of correlation among all values in numbers...
idx, idy = np.where(pd.isnull(data))    # To find the index of null values...

for col in data:            # Method for one-hot encoding of columns...
    data[col].replace({"Yes" : 1, "No" : 0}, inplace=True)
# Converting categorical data into one-hot dummies (One-Hot Encoding)...
dataset = pd.get_dummies(data=data, columns=["InternetService", "Contract", "PaymentMethod"])
dataset['Contract'].value_counts()    # It gives the unique value along with their counts as well...

RandomSampling = pd.concat([merged, data], axis=0)    # Concatenating two datasets...
data.sample(5000, replace=True)                   # Over Sampling- creating duplicates...