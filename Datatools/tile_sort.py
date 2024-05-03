import pandas as pd
from sklearn.cluster import KMeans
import os
import shutil

# 1. Indlæs CSV-dokumentet
df = pd.read_csv('Data/blandet.csv')

# 2. Udfør K-means clustering
# Antag at 'Hue', 'Saturation', og 'Value' er dine features
X = df[['Hue', 'Saturation', 'Value']]
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
df['Cluster'] = kmeans.labels_

# 3. Opret mapper og kopier filer
source_dir = 'Data/KD train tiles/blandet'
for i in range(7):
    os.makedirs(f'Data/KD train tiles/category {i}', exist_ok=True)

for index, row in df.iterrows():
    file_path = os.path.join(source_dir, row['Filename'])
    destination_dir = f'Data/KD train tiles/category {row["Cluster"]}'
    shutil.copy(file_path, destination_dir)

print("Filer er blevet fordelt i kategorimapper baseret på clustering.")
