# Just a script to reformat the csv from label studio
import pandas as pd
from pathlib import Path

og_df = pd.read_csv(Path(__file__).parent / "lstudio_out.csv", index_col=0)
print(len(og_df))
# Drop useless columns
df = og_df.drop(["annotator", "annotation_id", "created_at", "updated_at", "lead_time"], axis=1)

# Remove rows where choice is empty or "Invalid"

df = df[~((df["choice"] == "Invalid") | df["choice"].isnull())]

df.to_csv(Path(__file__).parent / "dataset_v1.csv")