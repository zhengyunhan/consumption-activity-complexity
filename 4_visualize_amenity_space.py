from ecomplexity import ecomplexity
from ecomplexity import proximity
from ecomplexity import calc_proximity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from util.util import weighted_mean, weighted_median
import networkx as nx
import plotly.graph_objects as go
from matplotlib import cm
import matplotlib.patches as mpatches
import importlib
from util.plot_network import  normalize_pci, get_node_colors

input_folder='complexity'
dirc='/output/'+input_folder+'/csv/full/disaggregate_2019.csv'
data=pd.read_csv(dirc)

rca_df = data.pivot(index='st_county_code', columns='CATEGORY_TAGS', values='rca')
rca_np=rca_df.to_numpy()

unbiq=data.groupby('CATEGORY_TAGS').agg('mean')['ubiquity'].reset_index()
pci=data.groupby('CATEGORY_TAGS').agg('mean')['pci'].reset_index()
unbiq_li=list(unbiq['ubiquity'])
tag_category=pd.read_csv('/data/aux/tag_category.csv')

category_data=unbiq.merge(tag_category,on='CATEGORY_TAGS').merge(pci,on='CATEGORY_TAGS')

proximity_matrix=calc_proximity.calc_continuous_proximity(rca_np,unbiq_li)

# Create a graph from the proximity matrix
def plot_network(proximity_matrix,category_data,label_list):
    unique_categories,category_to_color,node_colors = get_node_colors(category_data,proximity_matrix)

    G = nx.Graph()

    for i in range(proximity_matrix.shape[0]):
        G.add_node(i)

    mst = nx.maximum_spanning_tree(nx.Graph(proximity_matrix))

    edges = [(i, j, proximity_matrix[i, j]) for i in range(proximity_matrix.shape[0]) for j in range(i+1, proximity_matrix.shape[0])]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)

    G.add_edges_from(mst.edges(data=True))

    added_edges = set()
    count = 0

    for i, j, weight in edges:

        if (i, j) not in added_edges and (j, i) not in added_edges:
            G.add_edge(i, j, weight=weight)
            added_edges.add((i, j))
            count += 1
            # Stop if 1200 edges is reached
            if count >= 1200: 
                break

    pos = nx.spring_layout(G, k=0.1,iterations=50, weight='weight', seed=42)

    degree_dict = dict(G.degree())
    sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    all_nodes = {node for node, degree in sorted_nodes}


    labels = {node: category_data['CATEGORY_TAGS'][node] for node in all_nodes if category_data['CATEGORY_TAGS'][node] in label_list}
    select_nodes=[node for node in all_nodes if category_data['CATEGORY_TAGS'][node] in label_list]
    node_sizes = normalize_pci(category_data['pci'].values,scale=500)


    plt.figure(figsize=(12,12))

    other_nodes = set(all_nodes) - set(select_nodes)

    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=[node_sizes[n] for n in other_nodes],
                           node_color=[node_colors[n] for n in other_nodes], linewidths=1, edgecolors='white')
    
    nx.draw_networkx_nodes(G, pos, nodelist=select_nodes, node_size=[node_sizes[n] for n in select_nodes],
                           node_color=[node_colors[n] for n in select_nodes], linewidths=1.5, edgecolors='red')

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='grey')

    texts = []
    for node, (x, y) in pos.items():
        if node in labels:
            text = plt.text(x-0.1, y+0.06, labels[node], fontsize=18)
            texts.append(text)

    legend_handles = []
    for cat in unique_categories:
        legend_handles.append(mpatches.Patch(color=category_to_color[cat], label=f'{cat}'))

    plt.legend(handles=legend_handles, frameon=False, loc='center',fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#Plot amenity space

sort_df=category_data.sort_values('pci').reset_index(drop=True)
label_list=sort_df['CATEGORY_TAGS'][-10:].values
plot_network(proximity_matrix,category_data,label_list)

