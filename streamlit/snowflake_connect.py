import snowflake.connector
import pandas as pd

def load_reviews():
    conn = snowflake.connector.connect(
        user="FUCK",
        password="Pikitpagapong174511",
        account="IHIHVSG-YIC74668",  # <-- corrected
        warehouse="COMPUTE_WH",
        database="AVALANCHE_DB",
        schema="AVALANCHE_SCHEMA"
    )

    query = "SELECT * FROM reviews_with_sentiment;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df