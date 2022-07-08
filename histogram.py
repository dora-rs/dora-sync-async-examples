import matplotlib.pyplot as plt
import pandas as pd

# import jtplot submodule from jupyterthemes
from jupyterthemes import jtplot

# currently installed theme will be used to
# set plot style if no arguments provided
jtplot.style()
link = "http://localhost:16686/api/traces?service=sender&lookback=6h&prettyPrint=true&limit=10"

df_raw = pd.read_json(link)

l = []
for i in range(len(df_raw)):
    df_process = pd.DataFrame(df_raw.data[i]["processes"])
    df_ = pd.DataFrame(df_raw.data[i]["spans"])
    df_["processID"] = df_["processID"].map(df_process.T["serviceName"])
    l.append(df_)

df = pd.concat(l)

## In async thread
df_async = df[df["operationName"] == "in_async_thread"]
print("Total Response time:")
print(df_async.groupby(by="processID")["duration"].median())

grouped = df_async["duration"].groupby(by=df_async["processID"])

for group in grouped:
    plt.hist(
        group[1].values, label=group[0] + " resp. time", alpha=0.5, bins=50
    )
plt.legend(loc="upper right")
plt.title("Response Time")
plt.show()
# plt.title("async thread")
## In sync thread
df_sync = df[df["operationName"] == "in_sync_thread"]

print("Sync Median Duration:")
print(df_sync.groupby(by="processID")["duration"].median())
print("Sync Total Duration:")
print(df_sync.groupby(by="processID")["duration"].sum())

grouped = df_sync["duration"].groupby(by=df_sync["processID"])

for group in grouped:
    plt.hist(
        group[1].values, label=group[0] + " comp. time", alpha=0.5, bins=50
    )
plt.legend(loc="upper right")
plt.title("Computation time")
plt.show()
