import pandas as pd

link = "http://localhost:16686/api/traces?service=sender&lookback=6h&prettyPrint=true&limit=1"

df = pd.read_json(link)
df = pd.DataFrame(df.data[0]["spans"])

df = df[df["operationName"] == "tokio-spawn"]
df.groupby(by="processID")["duration"].mean()

df["duration"].hist(by=df["processID"])

import matplotlib

matplotlib.pyplot.show()
