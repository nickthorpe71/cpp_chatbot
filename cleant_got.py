# import pandas with shortcut 'pd'
import pandas as pd  
  
# read_csv function which is used to read the required CSV file
data = pd.read_csv('data/Game_of_Thrones_Script.csv')
  
# drop function which is used in removing or deleting rows or columns from the CSV files
data.drop(["Release Date","Season","Episode","Episode Title"], inplace=True, axis=1)
  
# save to new csv file
data.to_csv('data/Game_of_Thrones_Script_clean.csv', index=False)