# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:19:46 2022

@author: praja
"""

# Q1 a)
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Network Analysis\connecting_routes.csv")

data1 = data.iloc[0:500,1:9]

g = nx.Graph()  # craeting a empty graph with no. nodes and no. edges

g = nx.from_pandas_edgelist(data1, source = 'AER', target = "KZN")  # takes source and destination nodes from the dataframe 

print(nx.info(g))        # it gives the information about no of edges, nodes and average degree.

# degree centrality

d = nx.degree_centrality(g)
print(d)

# MN airport has the maximium degree of centrality

pos = nx.spring_layout(g, k= 0.2)
nx.draw_networkx(g,pos, node_size = 10, node_color = 'blue') 

# closeness Centrality
c = nx.closeness_centrality(g)
print(c)
# MNl has the highest closeness centrality

# Betweenness Centrality

b = nx.betweenness_centrality(g)
print(b)

# MNL has the highest netweenness centrality

# Eigen vector Centrality 

e = nx.eigenvector_centrality(g)
print(e)

# FMM has the  highest eigrn vector centrality



# Cluster cofficient

clcoeff = nx.clustering(g)
print(clcoeff)
# BOG,BOY,BTK has the highest  cluster coefficient

# average clustering

ac = nx.average_clustering(g)
print(ac)
    
# Q1 b)
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

A = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Network Analysis\flight_hault.csv")
# add names to column of  the dataframe 
A.columns = 'ID',' name ','city', 'country','IATA_FAA','ICAO','Latitude','Longitude','Altitude','Time','dst','Tz database time'
A = A.iloc[0:500, 1:12] # taking first 500 rows only
A.isna().sum()

from sklearn.impute import SimpleImputer

mode_imputation = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')

A["IATA_FAA"] =pd.DataFrame(mode_imputation.fit_transform(A[['IATA_FAA']]))

A.isna().sum()

x = nx.Graph()      # graph stores nodes and edges with optimal data or attributes

x = nx.from_pandas_edgelist(A, source='IATA_FAA', target="ICAO") # takes source and destination nodes from the dataframe 

print(nx.info(x))      # it gives the information about no of edges, nodes and average degree.

# Degree of Centrality

d = nx.degree_centrality(x)
print(d)

pos = nx.spring_layout(x, k = 0.2)
nx.draw_networkx(x, pos, node_size = 10, node_color ='blue')

#closeness Centrality
c = nx.closeness_centrality(x)
print(c)

# Betweenness centrality

b = nx.betweenness_centrality(x)
print(b)

# eigen vector centrality 
 e = nx.eigenvector_centrality(x)
print(e)

# clustering Coefficient
clust = nx.clustering(x)
print(c)

# Average Clustering

a = nx.average_clustering(x)
print(a)
##########################################################################

# Q2) 

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Network Analysis\facebook.csv")

mat = np.matrix(data)
h = nx.from_numpy_matrix(mat)
nx.draw(h)


data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Network Analysis\instagram.csv")

mat = np.matrix(data1)
h = nx.from_numpy_matrix(mat)
nx.draw(h)

data3= pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Network Analysis\linkedin.csv")
mat = np.matrix(data3)
h = nx.from_numpy_matrix(mat)
nx.draw(h)

    
    

