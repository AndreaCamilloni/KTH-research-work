import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.spatial import Delaunay
import networkx as nx

from IPython.display import display, clear_output

import xml.etree.ElementTree as ET
from xml.dom import minidom



#- 0: AMBIGUOUS
#- 1: nonTIL_stromal
#- 2: other_nucleus
#- 3: sTIL
#- 4: tumor_any

super_classes = {0: 'AMBIGUOUS', 1: 'nonTIL_stromal', 2: 'other', 3: 'sTIL', 4: 'tumor'}



# Mask Regions
colors = {
    'outside_roi': (255, 255, 255, 25),  # White, semi-transparent
    'tumor': (255, 0, 0, 70),           # Red, mostly opaque
    'stroma': (0, 255, 0, 100),          # Green, mostly opaque
    'lymphocytic_infiltrate': (0, 0, 255, 100), # Blue, less opaque
    'necrosis_or_debris': (255, 255, 0, 100),   # Yellow, less opaque
    'glandular_secretions': (255, 0, 255, 100), # Magenta, medium transparency
    'blood': (0, 255, 255, 100),          # Cyan, more transparent
    'exclude': (128, 128, 128, 90),      # Gray, more transparent
    'metaplasia_NOS': (128, 0, 0, 40),    # Dark Red, transparent
    'fat': (128, 128, 0, 40),             # Olive, transparent
    'plasma_cells': (128, 0, 128, 40),    # Purple, semi-transparent
    'other_immune_infiltrate': (0, 128, 128, 20), # Teal, very transparent
    'mucoid_material': (0, 128, 0, 60),   # Dark Green, semi-transparent
    'normal_acinus_or_duct': (0, 0, 128, 80),     # Navy, transparent
    'lymphatics': (255, 165, 0, 80),     # Orange, transparent
    'undetermined': (255, 192, 203, 70), # Pink, more transparent
    'nerve': (64, 224, 208, 80),         # Turquoise, medium transparency
    'skin_adnexa': (210, 105, 30, 80),   # Chocolate, less opaque
    'blood_vessel': (220, 20, 60, 90),   # Crimson, less opaque
    'angioinvasion': (95, 158, 160, 100), # Cadet Blue, mostly opaque
    'dcis': (72, 61, 139, 100),           # Dark Slate Blue, mostly opaque
    'other': (47, 79, 79, 100)            # Dark Slate Gray, almost fully opaque
}


# CELLS mapping RGB and BGR
color_dict_rgb = {'AMBIGUOUS': (0, 0, 0), 'nonTIL_stromal': (0, 0, 255), 'other': (0, 255, 0), 'sTIL': (255, 255, 0), 'tumor': (255, 0, 0)}
color_dict = {
    'AMBIGUOUS': (0, 0, 0),        # Black remains the same in both RGB and BGR
    'nonTIL_stromal': (255, 0, 0), # Blue in BGR
    'other': (0, 255, 0),          # Green remains the same in both RGB and BGR
    'sTIL': (0, 255, 255),         # Cyan in BGR
    'tumor': (0, 0, 255)           # Red in BGR
}


def get_xml_root(path, xml_file):
    xml_file = path + xml_file + '.xml' if not xml_file.endswith('.xml') else path + xml_file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

# Function to convert XML to DataFrame
def load_xml_to_dataframe(path, xml_file):
    root = get_xml_root(path, xml_file)

    # Create an empty list to store the parsed data
    data = []
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    # Iterate over each object in the XML
    for obj in root.findall('.//object'):
        item = {}

        # Extract data from each tag and add it to the dictionary
        item['class'] = int(obj.find('name').text.split('.0')[0])
        item['type'] = super_classes[item['class']] # need to add possibility to deal with main/superclasses directly here

        # Extracting bounding box data
        bndbox = obj.find('bndbox')
        item['xmin'] = bndbox.find('xmin').text
        item['ymin'] = bndbox.find('ymin').text
        item['xmax'] = bndbox.find('xmax').text
        item['ymax'] = bndbox.find('ymax').text

        xc = (int(item['xmin']) + int(item['xmax'])) / 2
        yc = (int(item['ymin']) + int(item['ymax'])) / 2
        
        #if xc > 0 and yc > 0 and xc < width and yc < height:
            # Append the dictionary to the list
        data.append(item)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df


def display_image(img, plot_output):
    with plot_output:
        clear_output(wait=True)  # Clear the previous image
        #img = mpimg.imread(image_path)
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def draw_bboxes_on_image(image_path, bboxes_df):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Image not found")
        return

    # iterate dataframe
    for index, row in bboxes_df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']) 
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_dict[row['type']], 2)
        
    # Create a legend
    legend_height_per_item = 100
    legend_width = 1000
    start_y = image.shape[0] - (len(color_dict) * legend_height_per_item) - 10

    for i, (label, color) in enumerate(color_dict.items()):
        end_y = start_y + legend_height_per_item
        cv2.rectangle(image, (image.shape[1] - legend_width, start_y), (image.shape[1], end_y), color, -1)
        cv2.putText(image, label, (image.shape[1] - legend_width + 10, start_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0) if color != (0,0,0) else (255,255,255), 3)
        start_y = end_y    


    
    # Display the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def draw_bboxes_on_masked_image(image, bboxes_df):
    # Load the image
    #image = cv2.imread(image_path)

    # iterate dataframe
    for index, row in bboxes_df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']) 
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_dict[row['type']], 2)
        
     # Create a legend
    legend_height_per_item = 20
    legend_width = 200
    start_y = image.shape[0] - (len(color_dict) * legend_height_per_item) - 10

    for i, (label, color) in enumerate(color_dict.items()):
        end_y = start_y + legend_height_per_item
        cv2.rectangle(image, (image.shape[1] - legend_width, start_y), (image.shape[1], end_y), color, -1)
        cv2.putText(image, label, (image.shape[1] - legend_width + 5, start_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        start_y = end_y    
    
    
    return image


def apply_colored_mask(image, mask, value, rgba, masks_df):
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask == int(masks_df[masks_df.label == value].GT_code.iloc[0])] = rgba[:3]  # RGB part for color
    #mask_overlay = (mask == int(masks_df[masks_df.label == value].GT_code)).astype(np.uint8) * rgba[3]  # Alpha part for transparency
    #return cv2.addWeighted(overlay, 1, image, 1, 0, mask=mask_overlay)

    alpha_mask = (mask == int(masks_df[masks_df.label == value].GT_code.iloc[0])).astype(float) * (rgba[3] / 255.0)  # Normalized Alpha

    # Manual blending
    return (image * (1 - alpha_mask[..., None]) + overlay * alpha_mask[..., None]).astype(np.uint8)


def apply_colored_mask_on_image(image_path, mask_path, masks_df):
    # Load the image and the mask
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, 0)

    # Convert the image to RGBA
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    # Apply the mask to the alpha channel
    alpha_channel = img_rgba[:, :, 3]
    unique_values = np.unique(mask)

    # Apply each colored mask
    for value, rgba in colors.items():
        img = apply_colored_mask(img, mask, value, rgba, masks_df)

    present_labels = []
    for value in np.unique(mask):
        if value in masks_df['GT_code'].values:
            rgba = colors[masks_df[masks_df['GT_code'] == value]['label'].iloc[0]]
            #img = apply_colored_mask(img, mask, value, rgba, masks_df)
            present_labels.append((masks_df[masks_df['GT_code'] == value]['label'].iloc[0], rgba))

    # Create a legend for the present labels
    legend_height_per_item = 100
    legend_width = 1000
    start_y = img.shape[0] - (len(present_labels) * legend_height_per_item) - 10

    for label, rgba in present_labels:
        end_y = start_y + legend_height_per_item
        cv2.rectangle(img, (img.shape[1] - legend_width, start_y), (img.shape[1], end_y), rgba[:3], -1)
        cv2.putText(img, label, (img.shape[1] - legend_width + 10, start_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0) if rgba != (0,0,0) else (255,255,255), 3)
        start_y = end_y
 

    return img





node_colors = {0: 'black', 1 : 'blue', 2: 'yellow', 3: 'green', 4: 'red'}

node_mask_colors = {
    0: (0, 0, 0),             # Black for 'outside_roi'
    1: (255, 0, 0),           # Red for 'tumor'
    2: (0, 255, 0),           # Green for 'stroma'
    3: (0, 0, 255),           # Blue for 'lymphocytic_infiltrate'
    4: (255, 255, 0),         # Yellow for 'necrosis_or_debris'
    5: (255, 0, 255),         # Magenta for 'glandular_secretions'
    6: (0, 255, 255),         # Cyan for 'blood'
    7: (192, 192, 192),       # Gray for 'exclude'
    8: (128, 0, 0),           # Maroon for 'metaplasia_NOS'
    9: (128, 128, 0),         # Olive for 'fat'
    10: (0, 128, 0),          # Dark Green for 'plasma_cells'
    11: (128, 0, 128),        # Purple for 'other_immune_infiltrate'
    12: (0, 128, 128),        # Teal for 'mucoid_material'
    13: (0, 0, 128),          # Navy for 'normal_acinus_or_duct'
    14: (255, 165, 0),        # Orange for 'lymphatics'
    15: (255, 20, 147),       # Deep Pink for 'undetermined'
    16: (75, 0, 130),         # Indigo for 'nerve'
    17: (173, 216, 230),      # Light Blue for 'skin_adnexa'
    18: (240, 128, 128),      # Light Coral for 'blood_vessel'
    19: (210, 105, 30),       # Chocolate for 'angioinvasion'
    20: (60, 179, 113),       # Medium Sea Green for 'dcis'
    21: (255, 228, 225)       # Misty Rose for 'other'
}

def normalize_tuple(tuple):
    return (tuple[0]/255,tuple[1]/255,tuple[2]/255)


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def generate_graph_from_bboxes(bboxes_df, mask_path):

    mask = cv2.imread(mask_path, 0)

    #get mask size width and height
    mask_width = mask.shape[1]
    mask_height = mask.shape[0]

    #drop bboxes outside mask
    bboxes_df['xc'] = (bboxes_df['xmin'].astype(int) + bboxes_df['xmax'].astype(int)) / 2
    bboxes_df['yc'] = (bboxes_df['ymin'].astype(int) + bboxes_df['ymax'].astype(int)) / 2
    bboxes_df = bboxes_df[(bboxes_df['xc'] > 0) & (bboxes_df['yc'] > 0) & (bboxes_df['xc'] < mask_width) & (bboxes_df['yc'] < mask_height)]

    # Assuming bboxes_df has columns ['xmin', 'ymin', 'xmax', 'ymax']
    # Calculate the center of each bounding box
    centers = np.vstack([(bboxes_df['xmin'].astype(int) + bboxes_df['xmax'].astype(int)) / 2, 
                         (bboxes_df['ymin'].astype(int) + bboxes_df['ymax'].astype(int)) / 2]).T

    # Perform Delaunay Triangulation
    tri = Delaunay(centers)

    # Create a graph from the triangulation
    graph = nx.Graph()

    # if couple of points have euclidean distance < 2 drop one of them
    #for i in range(len(centers)):
    #    for j in range(i+1, len(centers)):
    #        if(euclidean_distance(centers[i], centers[j]) < 2):
    #            tri.simplices = np.delete(tri.simplices, np.where(tri.simplices == i)[0][0], axis=0)
    #            centers = np.delete(centers, i, axis=0)
    #            break

    for i, point in enumerate(centers):
        #if point[0] < 0 or point[1] < 0:
        #    print("Error: Negative coordinates found")
        #    return
        #if point[0] >= mask.shape[1] or point[1] >= mask.shape[0]:
        #    print(point)
        #    print(bboxes_df.iloc[i])
        #    print("Error: Coordinates out of bounds")
        #    return
        mask_value = mask[int(point[1]), int(point[0])]
        graph.add_node(i, pos=tuple(point), label=bboxes_df['class'].iloc[i], gt = bboxes_df['type'].iloc[i],
                       cell_color=color_dict_rgb[bboxes_df['type'].iloc[i]], 
                       mask_value=int(mask_value), mask_color=node_mask_colors[int(mask_value)])

    # Add edges from the Delaunay triangulation
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                #if(simplex[i]==0 or simplex[j]==0):
                    #print(simplex[i], simplex[j])
                    #pass

                graph.add_edge(simplex[i], simplex[j], euclidean_distance=euclidean_distance(centers[simplex[i]], centers[simplex[j]]))

    
    return graph


def draw_graph_on_image(image_path, mask_path, bboxes_df, node_size=30, line_thickness=8, node_color='cell_color'):
    # Load the image
    image = cv2.imread(image_path)

    graph = generate_graph_from_bboxes(bboxes_df, mask_path)

    # Loop through the edges in the graph to draw lines
    for edge in graph.edges():
        pt1 = (graph.nodes[edge[0]]['pos'][0].astype(int), graph.nodes[edge[0]]['pos'][1].astype(int))
        pt2 = (graph.nodes[edge[1]]['pos'][0].astype(int), graph.nodes[edge[1]]['pos'][1].astype(int)) 
        cv2.line(image, pt1, pt2, (0, 0, 0), thickness=line_thickness)  # Blue color for lines

    # Loop through the nodes in the graph to draw circles
    for node in graph.nodes():
        cv2.circle(image, (int(graph.nodes[node]['pos'][0]),int(graph.nodes[node]['pos'][1])), node_size, graph.nodes[node][node_color], -1)  # Green color for nodes

    return image 

def save_graph_to_csv(graph, output_path, file_name):
    # Collect edge data in a list of dictionaries
    edge_data = [{'source': edge[0], 'target': edge[1], 'distance': graph.edges[edge]['euclidean_distance']}
                 for edge in graph.edges()]
    # Create a DataFrame and save to CSV
    pd.DataFrame(edge_data).to_csv(output_path + file_name + '_edges.csv', index=False)
        

    # Collect node data in a list of dictionaries
    node_data = [{'id': node, 'x': graph.nodes[node]['pos'][0], 'y': graph.nodes[node]['pos'][1], 
                  'gt': graph.nodes[node]['gt'],'class' : graph.nodes[node]['label'], 'mask': graph.nodes[node]['mask_value']}
                 for node in graph.nodes()]
    # Create a DataFrame and save to CSV
    pd.DataFrame(node_data).to_csv(output_path + file_name + '_nodes.csv', index=False)

def generate_tiles(wsi_name, image_path, mask_path, xml_path, fill = True,img_size = (2000,2000)):
    save_dir_img = os.path.join(image_path, "tiles")
    save_dir_mask = os.path.join(mask_path, "tiles")

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    if not os.path.exists(save_dir_mask):
        os.makedirs(save_dir_mask)
    if not os.path.exists(os.path.join(xml_path, 'tiles')):
        os.makedirs(os.path.join(xml_path, 'tiles'))
    image_path = image_path + wsi_name + '.png' 
    mask_path = mask_path + wsi_name + '.png'
    #xml_path = xml_path 


    bboxes_df = load_xml_to_dataframe(xml_path, wsi_name)
    
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)


    width, height = img.shape[1], img.shape[0]
    w_mod, h_mod = width % img_size[0], height % img_size[1]
    w_div, h_div = width // img_size[0], height // img_size[1]

    print(f"Image size: {width}x{height}")
    print(f"Tile size: {img_size[0]}x{img_size[1]}")
    print(f"Number of tiles: {w_div}x{h_div}")

    if fill:
        if w_mod > img_size[0] // 2:
            w_div += 1
        if h_mod > img_size[1] // 2:
            h_div += 1

    print(f"Number of tiles (after fill): {w_div}x{h_div}")    

     
    for i in range(w_div):
        for j in range(h_div):
            x_start = i * img_size[0]
            y_start = j * img_size[1]
            x_end = min(x_start + img_size[0], width) if i < w_div - 1 else width
            y_end = min(y_start + img_size[1], height) if j < h_div - 1 else height

            tile = img[y_start:y_end, x_start:x_end]
            tile_mask = mask[y_start:y_end, x_start:x_end]

            # Additional processing can be done here, e.g., using the mask

            tile_filename = f"{wsi_name}_tile_{i}_{j}"
            cv2.imwrite(os.path.join(save_dir_img, tile_filename + '.png'), tile)
            cv2.imwrite(os.path.join(save_dir_mask, tile_filename + '.png'), tile_mask)

            # save xml file
            bboxes_df_tile = filter_bboxes(bboxes_df, x_start,y_start,x_end,y_end)
            save_xml_for_tile(bboxes_df_tile, os.path.join(xml_path, 'tiles', tile_filename + '.xml'), img_size)

    



            #print(f"Saved tile {i}_{j} to {save_dir}")



def save_xml_for_tile(bboxes, tile_filename, tile_size, folder='YOLOv5', path_prefix='/yolo_workspace/data/bcss/whole_slides/'):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = tile_filename
    ET.SubElement(root, "path").text = path_prefix + tile_filename

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(tile_size[0])
    ET.SubElement(size, "height").text = str(tile_size[1])
    ET.SubElement(size, "depth").text = "3"

    #ET.SubElement(root, "segmented").text = "0"

    for i, bbox in bboxes.iterrows():
         
        obj = ET.SubElement(root, "object")
         
        ET.SubElement(obj, "name").text = str(bbox['class']) 
        #ET.SubElement(obj, "pose").text = "Unspecified"
        #ET.SubElement(obj, "truncated").text = "0"
        #ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(bbox['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(bbox['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(bbox['ymax'])

    #save 
    root = ET.ElementTree(root)
    root.write(tile_filename, encoding='utf-8', xml_declaration=True)

def filter_bboxes(bboxes_df, x_start,y_start,x_end,y_end):
    bboxes_df['xmin'] = bboxes_df['xmin'].astype(int)  
    bboxes_df['ymin'] = bboxes_df['ymin'].astype(int)  
    bboxes_df['xmax'] = bboxes_df['xmax'].astype(int)  
    bboxes_df['ymax'] = bboxes_df['ymax'].astype(int)  

    bboxes_df = bboxes_df[(bboxes_df['xmin'] >= x_start) & (bboxes_df['ymin'] >= y_start) & (bboxes_df['xmax'] <= x_end) & (bboxes_df['ymax'] <= y_end)]
    bboxes_df['xmin'] = bboxes_df['xmin'] - x_start
    bboxes_df['ymin'] = bboxes_df['ymin'] - y_start
    bboxes_df['xmax'] = bboxes_df['xmax'] - x_start
    bboxes_df['ymax'] = bboxes_df['ymax'] - y_start
    return bboxes_df

def tile_graphs_to_wsi_graph(wsi_name, tiles_path, prediction_path):
    tiles = os.listdir(tiles_path)
    tiles = [tile for tile in tiles if tile.startswith(wsi_name)]
    tiles = [tile for tile in tiles if tile.endswith('_nodes.csv')]

    # get rows and cols
    rows = []
    cols = []
    
    for tile in tiles:
        tile = tile.split('_tile_')[1]
        #print(tile)
        tile = tile.split('_')
        rows.append(int(tile[0]))
        cols.append(int(tile[1]))

    print(f"Number of tiles: {len(tiles)}")
    print(f"Number of rows: {len(set(rows))}")
    print(f"Number of cols: {len(set(cols))}")

    for i in len(set(rows)):
        for j in len(set(cols)):
            tile = f"{wsi_name}_tile_{i}_{j}"
            tile_nodes = pd.read_csv(os.path.join(tiles_path, tile + '_nodes.csv'))
            #tile_edges = pd.read_csv(os.path.join(tiles_path, tile + '_edges.csv'))
            
            #TODO: read node predictions and update dataframe
            tile_nodes['prediction'] = 0
            
