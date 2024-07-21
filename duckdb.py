
import duckdb

# connect to the database
con = duckdb.connect(":memory:")

# read the data from the CSV file
df = con.execute("SELECT * FROM 'aichatbot/resources/dataset.csv'").fetchdf()

# print the first 5 rows of the dataframe
print(df.head())

# close the connection
con.close()