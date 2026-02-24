import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/nifty50_step1_processed.csv")

plt.figure(figsize=(10,5))
plt.plot(df["logret"])
plt.xlabel("Time")
plt.ylabel("Log Return")
plt.title("NIFTY 50 Log Returns")
plt.grid(True)
plt.savefig("outputs/log_returns_plot.png", dpi=300)
plt.close()

print("Return plot saved to outputs/log_returns_plot.png")