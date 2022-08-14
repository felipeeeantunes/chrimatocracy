import folderstats

df = folderstats.folderstats("../data/", ignore_hidden=True)
df.to_csv("../data/prepared/folder_stats.csv")


df = folderstats.folderstats("../raw_data/", ignore_hidden=True)
df.to_csv("../data/raw/folder_stats.csv")
