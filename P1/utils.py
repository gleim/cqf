# utils.py
# Data Storage - SQLite
import pandas as pd
from quantmod.markets import getData
from sqlalchemy import create_engine, text


def get_stock_info():
    stock_table = pd.read_csv("../data/ind_nifty50list.csv")
    return stock_table


symbols = get_stock_info().Symbol.to_list()
nifty50 = [symbol + ".NS" for symbol in symbols]

# Connect to SQLite database
engine = create_engine("sqlite:///../db/equities.db", echo=False)


# SQL query to create table if not exists
def create_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    );
    """

    # Execute the table creation
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        print("Table created successfully.")


# Uncomment to run once and create the table
# create_table()

# -----------------------------------------
# Fetch and Store Function


# Fetch data and store in SQLite
def fetch_and_store_all():
    for symbol in nifty50:
        try:
            (
                getData(symbol, start_date="2020-01-01", end_date="2024-12-31")
                .reset_index()
                .assign(ticker=symbol.replace(".NS", ""))
                .loc[:, ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
                .rename(columns=str.lower)
                .to_sql("stock_data", engine, if_exists="append", index=False)
            )
            print(f"Saved: {symbol}")
        except Exception as e:
            print(f"Error with {symbol}: {e}")


# Uncomment to run once and populate the database
# fetch_and_store_all()


# -----------------------------------------
# Query: All Stocks
def query_all_stocks(engine=engine):
    query = text("""
        SELECT ticker, date, close
        FROM stock_data
        ORDER BY ticker, date
    """)

    with engine.connect() as conn:
        data = pd.read_sql(query, conn)

    # Preprocess
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)

    # Pivot: rows = date, columns = ticker, values = close
    data = data.pivot(columns="ticker", values="close")

    # Fill missing values for JIOFIN
    data["JIOFIN"] = data["JIOFIN"].ffill().bfill()

    # Remove 'ETERNAL' from the DataFrame
    data = data.drop(columns="ETERNAL", errors="ignore")

    return data


# -----------------------------------------
# Query from Database
def query_stock(ticker, engine=engine):
    query = text("SELECT * FROM stock_data WHERE ticker = :ticker ORDER BY date")

    with engine.connect() as conn:
        data = pd.read_sql(query, conn, params={"ticker": ticker})

    # Convert the date column to datetime format
    data["date"] = pd.to_datetime(data["date"])

    # Set the date column as the DataFrame index
    data.set_index("date", inplace=True)

    return data
