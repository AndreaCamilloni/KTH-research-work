import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import yaml

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom



def convert_patch_labels_to_original_image_labels(labels, row, class_names):
  
    # De-normalize the coordinates to patch coordinates
    labels['x_denormalized'] = labels['x'] * (row['xmax'] - row['xmin']) #df_img.iloc[0]['slice']
    labels['y_denormalized'] = labels['y'] * (row['ymax'] - row['ymin']) #df_img.iloc[0]['slice']
    labels['w_denormalized'] = labels['w'] * (row['xmax'] - row['xmin']) #df_img.iloc[0]['slice']
    labels['h_denormalized'] = labels['h'] * (row['ymax'] - row['ymin']) #df_img.iloc[0]['slice']
    
    # Convert to original image coordinates
    labels['x_original'] = labels['x_denormalized'] + row['xmin']
    labels['y_original'] = labels['y_denormalized'] + row['ymin']
    labels['w_original'] = labels['w_denormalized']
    labels['h_original'] = labels['h_denormalized']
    
    # Convert labels['class'] to class names
    labels['class'] = labels['class'].apply(lambda x: class_names[x])    

    return labels


# Convert to x_min, y_min, x_max, y_max
def convert_to_x_min_y_min_x_max_y_max(labels):
    labels['x_min'] = labels['x_original'] - labels['w_original']/2
    labels['y_min'] = labels['y_original'] - labels['h_original']/2
    labels['x_max'] = labels['x_original'] + labels['w_original']/2
    labels['y_max'] = labels['y_original'] + labels['h_original']/2
    return labels


        

def convert2xml(img1, df_img1_labels,img_info): 

    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = 'YOLOv5'
    filename = SubElement(annotation, 'filename')
    filename.text = img1
    path = SubElement(annotation, 'path')
    path.text = img1
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_info.W)#patches_info[patches_info['name'] == img1]['W'].values[0]
    height = SubElement(size, 'height')
    height.text = str(img_info.H)#patches_info[patches_info['name'] == img1]['H'].values[0]
    depth = SubElement(size, 'depth')
    depth.text = str(3)
    segmented = SubElement(annotation, 'segmented')
    segmented.text = str(0)

    for index, row in df_img1_labels.iterrows():
        object = SubElement(annotation, 'object')
        name = SubElement(object, 'name')
        name.text = str(row['class'])
        pose = SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = SubElement(object, 'difficult')
        difficult.text = str(0)
        bndbox = SubElement(object, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(int(row['x_min']))
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(int(row['y_min']))
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(int(row['x_max']))
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(int(row['y_max']))


    rough_string = tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    xml = reparsed.toprettyxml(indent="  ")
    return xml


class XMLannotations:
    def __init__(self, path, exp_name, output, data_yaml):
        self.path = path
        self.exp_name = exp_name
        self.output = output
        self.data_yaml = os.path.join(path, data_yaml)


    def __run__(self):
        # Read the data.yaml file
        # Read patches_info.csv
        # Read all the labels in self.path/self.exp_name/test/labels
        # Populate self.all_labels

        # Read yaml file
        with open(self.data_yaml) as file:
            self.yaml_doc = yaml.full_load(file)
            self.data_folder = self.yaml_doc['train'].split('/')[0]
            self.class_names = self.yaml_doc['names']
            
        # Read patches_info.csv
        self.patches_info = pd.read_csv(os.path.join(self.data_folder, 'patches_info.csv'))

        # Read all the labels in self.path/self.exp_name/test/labels
        self.all_labels = self.__get_all_labels__()

        # Create a folder to save the xml files
        self.__create_xml_folder__()

        # Convert all the labels to xml
        self.__transform_to_xml__()

        # Save the xml annotations in the folder self.output/self.exp_name/xml_annotations
        self.__save_xml__()


    def __transform_to_xml__(self):
        # Convert all the labels to xml
        for labels in self.all_labels:
            xml = convert2xml(labels['name'].iloc[0], labels, self.patches_info[self.patches_info['name']==labels['name'].iloc[0]].iloc[0])
            with open(os.path.join(self.xml_path, labels['name'].iloc[0].split('.')[0] + '.xml'), 'w') as f:
                f.write(xml)
                

    def __get_all_labels__(self):
        # Read all the labels in self.path/self.exp_name/test/labels
        # Filter out train and val labels from patches_info.csv
        self.patches_info = self.patches_info[self.patches_info['path'].split('\\')[0] =='test']
        #labels_df = labels_df[labels_df['path'] == 'new_data_processed_1\\test'] #TODO
        
        images_name = self.patches_info['name'].unique()
        self.all_labels = []
        for img in images_name:
            df_img = self.patches_info[self.patches_info['name'] == img]
            patches_labels = []
            for index, row in df_img.iterrows():
                # Read label file in yolo format
                label_path = os.path.join('..',row['path'], 'labels', row['filename'].split('.')[0] + '.txt')
                labels = pd.read_csv(label_path, sep=' ', header=None, names=['class', 'x', 'y', 'w', 'h'])
                labels = convert_patch_labels_to_original_image_labels(labels, row, self.class_names)
                labels = convert_to_x_min_y_min_x_max_y_max(labels)
                patches_labels.append(labels)
            patches_labels = pd.concat(patches_labels)
            patches_labels['name'] = img
            self.all_labels.append(patches_labels)
        return self.all_labels
        
    def __create_xml_folder__(self):
        # Create a folder to save the xml files
        self.xml_path = os.path.join(self.output, self.exp_name, 'xml_annotations')
        if not os.path.exists(self.xml_path):
            os.makedirs(self.xml_path)

if __name__ == "__main__":
    # Create the parser
    #my_parser = argparse.ArgumentParser(description='Converts YOLO annotations to XML')
    #my_parser.add_argument('Path', metavar='--path', type=str, help='the path to the folder containing the YOLO annotations', default='../yolov5/runs/')
    #my_parser.add_argument('Experiment Name', metavar='--exp-name', type=str, help='the name of the experiment', default='exp')
    #my_parser.add_argument('Output', metavar='--output', type=str, help='the path to the folder where the XML annotations will be saved', default='../PredictedLabels/')
    #my_parser.add_argument('Data yaml', metavar='--data-yaml', type=str, help='the path to the data.yaml file', default='../yolov5/data/data.yaml')
    # Execute the parse_args() method/
    #args = my_parser.parse_args()
    #print(args)

    path = '../new_data_processed_1/'
    exp_name = 'exp'
    output = './new_data_processed_1/'
    data_yaml = 'data.yaml'

    #path = '../new_data_processed_1/'
    
    #initiate the class
    xml_annotations = XMLannotations(path, exp_name, output, data_yaml)
    #run the class
    xml_annotations.__run__()

    
    


# Class to convert the predicted labels to xml annotations
# INPUT:
#   - path: the path to the folder containing the YOLO annotations (in runs/exp/test/{experiment_name}/labels)
#   - exp_name: the name of the experiment
#   - output: the path to the folder where the XML annotations will be saved
#   - data_yaml: the path to the data.yaml file => to get the data folder path where to read the file patches_info.csv

# OUTPUT:
#   - xml annotations in the output folder
#  - the xml annotations are saved in the folder output/exp/xml_annotations

# Example of use:
# python convert_yolo_to_xml.py --path ../yolov5/runs/ --exp-name exp --output ../PredictedLabels/ --data-yaml ../yolov5/data/data.yaml
