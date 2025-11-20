
import pandas as pd
import snowflake.connector


conn = snowflake.connector.connect(
    user='FUCK',
    password='Pikitpagapong174511',
    account='IHIHVSG-YIC74668',
    warehouse='COMPUTE_WH',
    database='AVALANCHE_DB',
    schema='AVALANCHE_SCHEMA',
)


query = "SELECT * FROM reviews_with_sentiment;"
df = pd.read_sql(query, conn)


conn.close()

print(df.head())

