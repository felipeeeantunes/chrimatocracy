import folderstats

df = folderstats.folderstats('../data/', ignore_hidden=True)
df.to_csv('../data/folder_stats.csv')


df = folderstats.folderstats('../raw_data/', ignore_hidden=True)
df.to_csv('../raw_data/folder_stats.csv')