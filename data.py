import yfinance as yf

df = yf.download("^NSEI", start="2010-01-01", end="2020-01-01")

print(df.head())   
df.to_csv("nifty50.csv")
