import argparse
import os
import json

from utils import *


def xml2png(xml_file):
    xml_file = xml_file.split(".xml")[0] + ".png"
    return xml_file

def png2xml(png_file):
    png_file = png_file.split(".png")[0] + ".xml"
    return png_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", default="configs.json")
    parser.add_argument("--xml_path", help="Path to XML files")
    parser.add_argument("--dataset_path", help="Path to dataset")
    parser.add_argument("--img_path", help="Path to image files")
    parser.add_argument("--mask_path", help="Path to mask files")
    parser.add_argument("--cell_class_type", help="Cell class type", choices=["superclass", "mainclass"])
    parser.add_argument("--output_path", help="Path to output files")
    parser.add_argument("--name_exp", help="Name of experiment")
    args = parser.parse_args()

    # Load config if specified
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            print(config)
        xml_path = config['xml_path']
        dataset_path = config.get('dataset_path')
        img_path = config['img_path']
        mask_path = config.get('mask_path')
        cell_class_type = config.get('cell_class_type')
        output_path = config.get('output_path') + config.get('name_exp')
    else:
        xml_path = args.xml_path
        dataset_path = args.dataset_path
        img_path = os.path.join(dataset_path, args.img_path)
        mask_path = os.path.join(dataset_path, args.mask_path)
        cell_class_type = args.cell_class_type
        output_path = args.output_path + args.name_exp

    img_path = dataset_path + img_path
    mask_path = dataset_path + mask_path


    #xml_path = args.xml_path
    #datasets_path = args.datasets_path
    #img_path = os.path.join(datasets_path, args.img_path)
    #mask_path = os.path.join(datasets_path, args.mask_path)
    #cell_class_type = args.cell_class_type # superclass or mainclass

    # Check if paths exist
    if not os.path.exists(xml_path):
        print("Error: XML path does not exist")
        return
    if not os.path.exists(img_path):
        print("Error: Image path does not exist")
        return
    if not os.path.exists(mask_path):
        print("Error: Mask path does not exist")
        return
    
    # check cell_class_type
    if cell_class_type != "superclass" and cell_class_type != "mainclass":
        print("Error: cell_class_type must be either superclass or mainclass")
        return
    
    # Check if XML files and images are the same
    xml_files = os.listdir(xml_path)
    img_files = os.listdir(img_path)
    mask_files = os.listdir(mask_path)
    xml_files.sort()
    img_files.sort()
    mask_files.sort()

    if len(xml_files) != len(img_files):
        print("Number of XML files: {}".format(len(xml_files)))
        print("Number of image files: {}".format(len(img_files)))
        print("Error: Number of XML files and images are not the same")
        #return
    if len(xml_files) != len(mask_files):
        print("Number of XML files: {}".format(len(xml_files)))
        print("Number of mask files: {}".format(len(mask_files)))
        print("Error: Number of XML files and masks are not the same")
        #return
    
    #print(xml2png(xml_files[0]))
    # Discard XML files that do not have a corresponding image
    xml_files = [f for f in xml_files if xml2png(f) in img_files]
    xml_files = [f for f in xml_files if xml2png(f) in mask_files]
    img_files = [f for f in img_files if png2xml(f) in xml_files]
    mask_files = [f for f in mask_files if png2xml(f) in xml_files]
    print("Number of XML files: {}".format(len(xml_files)))
    print("Number of image files: {}".format(len(img_files)))
    print("Number of mask files: {}".format(len(mask_files)))

    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + "nodes/")
        os.makedirs(output_path + "edges/")

    for i in range(len(xml_files)):
         
        #graph = nx.Graph()
        xml_file = xml_files[i]
        #img_file = xml2png(xml_file)
        mask_file = os.path.join(mask_path, xml2png(xml_file))
        print("Processing {}/{}: {}".format(i+1, len(xml_files), xml_file))
         
        # Read XML file
        bboxes_df = load_xml_to_dataframe(xml_path, xml_file)

        graph = generate_graph_from_bboxes(bboxes_df, mask_file)

         # check if entry pos of the nodes exists
        if not nx.get_node_attributes(graph, "pos"):
            print("Error: No node position found")
            return

        # Save graph
        save_graph_to_csv(graph, output_path, xml_file.split(".xml")[0])
      
    



if __name__ == "__main__":
    main()
