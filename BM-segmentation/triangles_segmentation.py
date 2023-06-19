import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import sys

import networkx as nx
from itertools import combinations

import argparse



def main(args):
    image_path = os.path.join(args.intelligraph_path, "slides/", args.tile_name + '.tif')

     
    task = 'Test/'
    edges = 'Datasets/'+ task + args.tile_name + '_delaunay_orig_forGraphSAGE_edges.csv'
    nodes = 'Datasets/'+ task + args.tile_name + '_delaunay_orig_forGraphSAGE_nodes.csv'
    
    json_annot = os.path.join(args.intelligraph_path, "edge_results", ("1013_ablation_n1_e1234_"+ args.tile_name + "_testing_edges_result.json"))
    # the file have this format: [[u, v, label], [u, v, label], ...]
    # where u and v are the nodes and label is the edge type
    # Read the json file
    pred = pd.read_json(json_annot)

    nodes_df, edges_df = read_data(nodes, edges)

    if args.prediction:
        for index, row in pred.iterrows():
            edges_df.loc[(edges_df.source == row[0]) & (edges_df.target == row[1]), 'type'] = 1
            edges_df.loc[(edges_df.source == row[1]) & (edges_df.target == row[0]), 'type'] = 1

    # Create the graph
    G = nx.Graph()
    #G.add_nodes_from(nodes,)
    for index, row in nodes_df.iterrows():
        G.add_node(row['id'], type=row['gt'], coords=(row['x'], row['y']))

    #G.add_edges_from(edges_df[['source', 'target']].values, )
    for u,v,d in edges_df[['source', 'target', 'type']].values:
        G.add_edge(u, v, label=d)

    # Graph plot properties
    pos = nodes_df[['x', 'y']].values
    gt = nodes_df['gt'].values
    color = {'epithelial': 'red', 'fibroblast and endothelial': 'blue', 'inflammatory': 'green', 'lymphocyte': 'yellow', 'apoptosis / civiatte body': 'black'}
    edge_color = {0:'black', 1: 'red'}#
    edge_types = edges_df['type'].values
    edge_colors = [edge_color[int(label)] for u, v, label in G.edges.data('label')]

    # Get Triangular structures
    triangle_df = find_triangles(G, edges_df, nodes_df)

    if not args.show_green:
                
        for _, row in triangle_df[triangle_df['count'] == 1].iterrows():
            #print(row['Node1'], row['Node2'], row['Node3'])
            FILTER, tri = get_neighbouring_triangles(row, triangle_df, edges_df)
            if FILTER:
                # update the count of the triangle to 2 and to 0 of the current triangle
                triangle_df.loc[(triangle_df['Node1'] == row['Node1']) & (triangle_df['Node2'] == row['Node2']) & (triangle_df['Node3'] == row['Node3']), 'count'] = 0
                triangle_df.loc[(triangle_df['Node1'] == tri['Node1'].values[0]) & (triangle_df['Node2'] == tri['Node2'].values[0]) & (triangle_df['Node3'] == tri['Node3'].values[0]), 'count'] = 2
            

    
    # Load tile image
    image = plt.imread(image_path)
    image = image[:, :, 0:3]

    dpi = 100
    height, width, depth = image.shape
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure and axis object
    fig, ax = plt.subplots( figsize=figsize)

    # Figure size 
    #fig = plt.figure(figsize=figsize)

    # Plot the image
    if not args.no_background:
        ax.imshow(image)

    # Filter out just triangles in the BM
    filtered_triangles = triangle_df[triangle_df['count'] > 0]

    for _, row in filtered_triangles.iterrows():
        x1, y1, x2, y2, x3, y3 = row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3']
        count = row['count']
        if count == 3:
            triangle_color = 'green'
        elif count == 2:
            triangle_color = 'yellow'
        else:
            triangle_color = 'red'
        triangle = patches.Polygon([(x1, y1), (x2, y2), (x3, y3)], closed=True, facecolor=triangle_color, alpha=0.5)
        ax.add_patch(triangle)

    # Set the limits of the plot
    x_min, x_max = triangle_df[['x1', 'x2', 'x3']].min().min() - 1, triangle_df[['x1', 'x2', 'x3']].max().max() + 1
    y_min, y_max = triangle_df[['y1', 'y2', 'y3']].min().min() - 1, triangle_df[['y1', 'y2', 'y3']].max().max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    nx.draw(G, pos, node_size=30, width= 0.5, node_color=[color[gt[i]] for i in range(len(gt))], edge_color=edge_colors )

    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # save the figure with original size
    plt.savefig(args.tile_name + '.png', dpi=dpi, bbox_inches='tight', transparent=True)

    

def read_data(nodes, edges):
    with open(edges, 'r') as f:
        first_line = f.readline()
        edges_cols = first_line.split(',')
        edges_cols = [col.strip() for col in edges_cols]
        edges_cols = [col.strip('"') for col in edges_cols]
    
    with open(nodes, 'r') as f:
        first_line = f.readline()
        nodes_cols = first_line.split(',')
        nodes_cols = [col.strip() for col in nodes_cols]
        nodes_cols = [col.strip('"') for col in nodes_cols]

    edges_df = pd.read_csv(edges, header=None, names=edges_cols)[1:]
    nodes_df = pd.read_csv(nodes, header=None, names=nodes_cols)[1:]

    edges_df['source'] = edges_df['source'].astype(int)
    edges_df['target'] = edges_df['target'].astype(int)
    edges_df['type'] = edges_df['type'].astype(int)
    edges_df['distance'] = edges_df['distance'].astype(float)

    nodes_df['id'] = nodes_df['id'].astype(int)
    nodes_df['lym'] = nodes_df['lym'].astype(float)
    nodes_df['epi'] = nodes_df['epi'].astype(float)
    nodes_df['fib'] = nodes_df['fib'].astype(float)
    nodes_df['inf'] = nodes_df['inf'].astype(float)
    nodes_df['x'] = nodes_df['x'].astype(float)
    nodes_df['y'] = nodes_df['y'].astype(float)

    return nodes_df, edges_df

def find_triangles(G, edges_df, nodes_df):
 
    # Find all triangles
    triangles = []
    #count_crossing_Edges = dict()
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            if G.has_edge(u, v):
                triangles.append(sorted([node, u, v]))
                #tmp = frozenset((node,u,v))
                #count = 0                
                #count_crossing_Edges[tmp] = count

    # Remove duplicate triangles
    unique_triangles = [list(x) for x in set(tuple(x) for x in triangles)]
    triangle_df = pd.DataFrame(unique_triangles, columns=['Node1', 'Node2', 'Node3'])
    #print(triangle_df)
    for idx, row in triangle_df.iterrows():
        node1_xy = nodes_df.loc[nodes_df['id'] == row['Node1'], ['x','y']].iloc[0].values
        node2_xy = nodes_df.loc[nodes_df['id'] == row['Node2'], ['x','y']].iloc[0].values
        node3_xy = nodes_df.loc[nodes_df['id'] == row['Node3'], ['x','y']].iloc[0].values

        triangle_df.loc[idx, ['x1','y1']] = node1_xy
        triangle_df.loc[idx, ['x2','y2']] = node2_xy
        triangle_df.loc[idx, ['x3','y3']] = node3_xy

        count = 0
        if edges_df[(edges_df['source'] == row['Node1']) & (edges_df['target'] == row['Node2']) | 
                (edges_df['source'] == row['Node2']) & (edges_df['target'] == row['Node1'])]['type'].values[0] == 1:
            count +=1
        if edges_df[(edges_df['source'] == row['Node1']) & (edges_df['target'] == row['Node3']) |
                    (edges_df['source'] == row['Node3']) & (edges_df['target'] == row['Node1'])]['type'].values[0] == 1:
            count +=1    
        if edges_df[(edges_df['source'] == row['Node2']) & (edges_df['target'] == row['Node3']) |
                    (edges_df['source'] == row['Node3']) & (edges_df['target'] == row['Node2'])]['type'].values[0] == 1:
            count +=1
        
        triangle_df.loc[idx, ['count']] = count
    
    return triangle_df

def get_crossing_edge(row, edges_df):
    #node1, node2, node3 = triangle['Node1'], triangle['Node2'], triangle['Node3']

    if edges_df[(edges_df['source'] == row['Node1']) & (edges_df['target'] == row['Node2']) | 
             (edges_df['source'] == row['Node2']) & (edges_df['target'] == row['Node1'])]['type'].values[0] == 1:
        return [row['Node1'], row['Node2']]
    if edges_df[(edges_df['source'] == row['Node1']) & (edges_df['target'] == row['Node3']) |
                (edges_df['source'] == row['Node3']) & (edges_df['target'] == row['Node1'])]['type'].values[0] == 1:
        return [row['Node1'], row['Node3']]   
    if edges_df[(edges_df['source'] == row['Node2']) & (edges_df['target'] == row['Node3']) |
                (edges_df['source'] == row['Node3']) & (edges_df['target'] == row['Node2'])]['type'].values[0] == 1:
        return [row['Node2'], row['Node3']]


def get_neighbouring_triangles(triangle, triangle_df, edges_df):
    edge = get_crossing_edge(triangle, edges_df)
    #print('Edge: ', edge)
    node1, node2 = edge[0], edge[1] 
    #nodea, nodeb, nodec = triangle['Node1'], triangle['Node2'], triangle['Node3']
    
    if edge is None:
        return None


    tri = triangle_df[(triangle_df['Node1'] == node1) & (triangle_df['Node2'] == node2) | 
                (triangle_df['Node2'] == node1) & (triangle_df['Node3'] == node2) | 
                (triangle_df['Node1'] == node1) & (triangle_df['Node3'] == node2) ]
    tri_1 = tri[tri['count'] == 1]
    tri = tri[tri['count'] == 3]
    if len(tri) == 0: # if there is no triangle with count 3
        return False, tri_1
    else:
        return True, tri



# Arguments parsing
if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Triangles BM segmentation')
    
    # Add the arguments
    my_parser.add_argument('--tile-name',
                           type=str,
                           help='the full name of the tile to be segmented',
                           default='P28_7_5')
    my_parser.add_argument('--intelligraph-path',
                            type=str,
                            help='the path to intelligraph folder',
                            default='../../IntelliGraph/')
    my_parser.add_argument('--no-background',
                            action='store_true',
                            help='Background image',
                            default=False)
    my_parser.add_argument('--prediction',
                            action='store_true',
                            help='Show predicted edges instead of ground truth edge', 
                            default=False)
    my_parser.add_argument('--show-green',
                            action='store_true',
                            help='Show green triangles if present', 
                            default=False)
    
    args = my_parser.parse_args()
    
    print(args)


    main(args)

