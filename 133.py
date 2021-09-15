import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

df = pd.read_csv('star_with_gravity.csv')

mass = df['Mass']
radius = df['Radius']

# new_fig = px.scatter(x=mass, y=radius)
# new_fig.show()

data = df.iloc[:, [3,4]]
#print(data)

wcss = []

for i in range(1, 11):
  kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
  kmeans.fit(data)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(range(1,11), wcss, marker='o', color='red')
plt.title("The Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()