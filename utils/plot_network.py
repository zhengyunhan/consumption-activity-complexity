import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from matplotlib import cm
import matplotlib.patches as mpatches
from adjustText import adjust_text


def normalize_pci(pci_values, new_min=5,scale=1):
    min_pci = np.min(pci_values)
    max_pci = np.max(pci_values)
    # Normalize and scale
    normalized = ((pci_values - min_pci)*scale / (max_pci - min_pci) + new_min)
    return normalized


def get_node_colors(category_data,proximity_matrix):
    # unique_categories = sorted(set(category_data['Category']), key=lambda x: x[0])
    unique_categories = ['Manufacturing','Administrative and Support and Waste Management and Remediation Services','Construction','Retail Trade',\
    'Information','Educational Services','Other Services (except Public Administration)','Professional, Scientific, and Technical Services','Management of Companies and Enterprises',\
    'Health Care and Social Assistance','Finance and Insurance','Real Estate and Rental and Leasing','Transportation and Warehousing',\
    'Public Administration','Accommodation and Food Services','Wholesale Trade','Arts, Entertainment, and Recreation','Utilities']
    colors = cm.get_cmap('tab20', len(unique_categories)) 
    category_to_color = {cat: colors(i) for i, cat in enumerate(unique_categories)}
    node_colors = [category_to_color[category_data['Category'][i]] for i in range(proximity_matrix.shape[0])]
    return unique_categories,category_to_color,node_colors

