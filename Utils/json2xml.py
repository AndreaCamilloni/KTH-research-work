import os
import json
import argparse
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import pandas as pd

def convert_coords(df):
    width, height = 32, 32
    df['xmin'] = df['x'] - width/2
    df['ymin'] = df['y'] - height/2
    df['xmax'] = df['x'] + width/2
    df['ymax'] = df['y'] + height/2

    df['class'] = df['groundTruthType']
    
    return df


def convert2xml(image_name, data,img_size = (2000, 2000)): 

    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = 'GT_cell_annotation'
    filename = SubElement(annotation, 'filename')
    filename.text = image_name
    path = SubElement(annotation, 'path')
    path.text = image_name
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_size[0])#patches_info[patches_info['name'] == img1]['W'].values[0]
    height = SubElement(size, 'height')
    height.text = str(img_size[1])#patches_info[patches_info['name'] == img1]['H'].values[0]
    depth = SubElement(size, 'depth')
    depth.text = str(3)
    segmented = SubElement(annotation, 'segmented')
    segmented.text = str(0)

    for index, row in data.iterrows():
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
        xmin.text = str(int(row['xmin']))
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(int(row['ymin']))
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(int(row['xmax']))
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(int(row['ymax']))


    rough_string = tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    xml = reparsed.toprettyxml(indent="  ")
    return xml

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='json', help='path to json files')
    parser.add_argument('--xml-path', type=str, default='xml', help='path to save xml files')
    return parser.parse_args()

def main():
    args = args()
    # read all json files in json_path
    json_path = args.json_path

    # save all xml files in xml_path
    xml_path = args.xml_path
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)

    for json_file in os.listdir(json_path):
        print(json_file)
        if json_file.endswith('.json'):
            json_file_path = os.path.join(json_path, json_file)
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df = convert_coords(df)
                tile_name = json_file.split('.')[0]
                xml = convert2xml(tile_name, df)
                xml_file_path = os.path.join(xml_path, tile_name + '.xml')
                with open(xml_file_path, 'w') as f:
                    f.write(xml)

if __name__ == '__main__':
    main()