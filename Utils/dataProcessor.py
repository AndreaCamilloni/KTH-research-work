import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import cv2
import argparse

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt




def xml_to_df(path):
    xml_list = []
    print("Parsing XML files...")
    print("Path: ", path)
    #print path in which is now
    print(os.getcwd())
    for xml_file in os.listdir(path):
        tree = ET.parse(os.path.join(path, xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def calculate_slice_bboxes(
    #image_height: int,
    #image_width: int,
    img,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as patches in xyxy format.
    :param img: image to be sliced
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of patches in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    # Add white pixels to make sure all patches have the same size and overlaps equally --> NEW FEATURE
    img = cv2.copyMakeBorder(img, 0, slice_height - img.shape[0] % slice_height - x_overlap, 0, slice_width - img.shape[1] % slice_width-y_overlap, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    image_height = img.shape[0]
    image_width = img.shape[1]

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

# Visualize image patch with given coordinates
def visualize_patch(img, slice):
    a = img.copy()
    # Crop the image
    a = a[slice[1]:slice[3], slice[0]:slice[2]]
    plt.imshow(a)
    plt.show()


# get dataframe of the patch such that xmin, ymin, xmax, ymax are into the given slice coordinates
def get_df_slice(df, slice):
    xmin = slice[0]
    ymin = slice[1]
    xmax = slice[2]
    ymax = slice[3]
    return df[(df['xmin'] >= xmin) & (df['ymin'] >= ymin) & (df['xmax'] <= xmax) & (df['ymax'] <= ymax)]

# Convert the coordinates of BBoxes to the patch coordinates
def convert_coordinates_to_patch(df_slice, slice):
    df_slice['xmin'] = df_slice['xmin'].apply(lambda x: x - slice[0]) 
    df_slice['ymin'] = df_slice['ymin'].apply(lambda x: x - slice[1])
    df_slice['xmax'] = df_slice['xmax'].apply(lambda x: x - slice[0])
    df_slice['ymax'] = df_slice['ymax'].apply(lambda x: x - slice[1])
    #df_slice['ymin'] = df_slice['ymin'] - slice[1]
    #df_slice['xmax'] = df_slice['xmax'] - slice[0]
    #df_slice['ymax'] = df_slice['ymax'] - slice[1]
    return df_slice

def convert_coordinates_to_original(df_slice, slice):
    df_slice['xmin'] = df_slice['xmin'] + slice[0]
    df_slice['ymin'] = df_slice['ymin'] + slice[1]
    df_slice['xmax'] = df_slice['xmax'] + slice[0]
    df_slice['ymax'] = df_slice['ymax'] + slice[1]
    return df_slice

# Given an image create the patches and the corresponding .txt files and return a daframe with the information (Original image sizes, patch sizes, patch coordinates, patch id)
def create_patches(img_filename, images_path, df, destination, class_to_num, slice_size=512, overlapping=0.2):
    df = df[df.filename == img_filename]
    # Read the image
    img = cv2.imread(os.path.join(images_path, img_filename))
    # Calculate the slices
    #slices = calculate_slice_bboxes(img.shape[0], img.shape[1], slice_size, slice_size, overlapping, overlapping)
    slices = calculate_slice_bboxes(img, slice_size, slice_size, overlapping, overlapping)

    # dataframe to store the information of the slices
    image_info = pd.DataFrame(columns=['name', 'filename',  'xmin', 'ymin', 'xmax', 'ymax', 'slice','path', 'W', 'H','Overlapping'])
   

    for i, slice in enumerate(slices):
        # Crop the image
        img_slice = img[slice[1]:slice[3], slice[0]:slice[2]]
        # Get the dataframe of the patch
        df_slice = get_df_slice(df, slice)
        # Convert the coordinates of BBoxes of the patch
        df_slice = convert_coordinates_to_patch(df_slice, slice)
        # Check for path existence
        #if not os.path.exists(os.path.join(destination,'patches')):
        #    os.makedirs(os.path.join(destination,'patches'))
        #    os.makedirs(os.path.join(destination,'patches/images'))
        #    os.makedirs(os.path.join(destination,'patches/labels'))
        
        #Save patch only if it contains at least 10 objects
        if len(df_slice) < 10:
            continue
        
        # Save the .txt file of the patch in yolo format <class> <x_center> <y_center> <width> <height>
        df_slice['class'] = df_slice['class'].apply(lambda x: class_to_num[x])
        df_slice['x_center'] = (df_slice['xmin'] + df_slice['xmax']) / 2 / slice_size
        df_slice['y_center'] = (df_slice['ymin'] + df_slice['ymax']) / 2 / slice_size
        df_slice['width'] = (df_slice['xmax'] - df_slice['xmin']) / slice_size
        df_slice['height'] = (df_slice['ymax'] - df_slice['ymin']) / slice_size
        df_slice = df_slice[['class', 'x_center', 'y_center', 'width', 'height']]

        name = os.path.splitext(img_filename)[0]
        df_slice.to_csv(f'{destination}/labels/{name}_{i}.txt', index=False, header=False, sep=' ')
        original_img_width = df[df['filename']==img_filename]['width'].values[0]
        original_img_height = df[df['filename']==img_filename]['height'].values[0]
        # Save the image
        cv2.imwrite(f'{destination}/images/{name}_{i}.tif', img_slice)
        # Store the information of the patch
        image_info = image_info.append({'name':img_filename ,'filename': f'{name}_{i}.tif', 'xmin': slice[0], 'ymin': slice[1], 'xmax': slice[2], 'ymax': slice[3], 'slice':slice_size, 'path': str(destination), 'W': original_img_width, 'H': original_img_height, 'Overlapping': overlapping}, ignore_index=True)


    return image_info

def yolo_format(df, class_to_num):
    df['class_num'] = df['class'].apply(lambda x: class_to_num[x])
    df['x_center'] = (df['xmin'] + df['xmax']) / 2 / df['width']
    df['y_center'] = (df['ymin'] + df['ymax']) / 2 / df['height']
    df['bbox_width'] = (df['xmax'] - df['xmin']) / df['width']
    df['bbox_height'] = (df['ymax'] - df['ymin']) / df['height']
    return df



    
class DataProcessor:
    def __init__(self, source, destination, slice_size=512, overlapping=0.2, slicing=False, val_split=0.2, train_imgs=[], val_imgs=[], test_imgs=[]):
        self.source = source
        self.destination = destination
        self.slice_size = slice_size
        self.overlapping = overlapping
        self.slicing = slicing

        self.annotations = os.path.join(self.source, 'annotations')
        self.images_path = os.path.join(self.source, 'images')

        i = 1
        self.destination = f'{self.destination}_{i}'
        while os.path.exists(self.destination):
            # If the directory already exists, append an incremental number
            self.destination = f'{destination}_{i}'
            print(f'Folder {self.destination} already exists')
            i += 1
        print(f'Creating folder {self.destination}...')
        print(os.path.exists(self.destination))
        os.mkdir(self.destination)

        self.images_path_out = os.path.join(self.destination, 'images')
        self.labels_path = os.path.join(self.destination, 'labels')

        self.val_split = val_split

        self.data = xml_to_df(self.annotations)
        self.classes = self.data['class'].unique()

        print('Founded classes: ', self.classes)
        if 'apoptosis / civiatte body' in self.classes:
            print('Dropping class "apoptosis / civiatte body"')
            # DROP 'apoptosis / civiatte body' class 
            self.data = self.data[self.data['class'] != 'apoptosis / civiatte body'] 
            self.classes = self.data['class'].unique()

        # Create a dictionary to map the class name to a number
        self.class_to_num = {}
        for i, c in enumerate(self.classes):
            self.class_to_num[c] = i
        
        print('Class to number mapping: ', self.class_to_num) 

        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        self.test_imgs = test_imgs

        # TODO Now every sets must be spefied in the config file otherwise the split_data function will be called
        if self.train_imgs == [] or self.val_imgs == [] or self.test_imgs == []:
            self.train_imgs, self.val_imgs, self.test_imgs = self.split_data()


        # Check if the images and the labels exist in the folders otherwise remove them from the list
        self.train_imgs = [img for img in self.train_imgs if self.check_existance_img_and_label(img, 'train')]
        self.val_imgs = [img for img in self.val_imgs if self.check_existance_img_and_label(img, 'val')]
        self.test_imgs = [img for img in self.test_imgs if self.check_existance_img_and_label(img, 'test')]

        # Print the number of samples for set and which one
        print(f'Training set {self.train_imgs} - Number of samples {len(self.train_imgs)}')
        print(f'Training set {self.val_imgs} - Number of samples {len(self.val_imgs)}')
        print(f'Training set {self.test_imgs} - Number of samples {len(self.test_imgs)}')
        #Update self.data keeping only the images that are in the sets
        self.data['name'] = self.data['filename'].apply(lambda x: os.path.splitext(x)[0])
        self.data = self.data[self.data['name'].isin(self.train_imgs + self.val_imgs + self.test_imgs)]


        # Create the images and labels folders train, val and test
        self.train_path = os.path.join(self.destination, 'train')
        self.val_path = os.path.join(self.destination, 'val')
        self.test_path = os.path.join(self.destination, 'test')

        os.mkdir(self.train_path)
        os.mkdir(self.val_path)
        os.mkdir(self.test_path)

        self.train_images_path = os.path.join(self.train_path, 'images')
        self.train_labels_path = os.path.join(self.train_path, 'labels')
        self.val_images_path = os.path.join(self.val_path, 'images')
        self.val_labels_path = os.path.join(self.val_path, 'labels')
        self.test_images_path = os.path.join(self.test_path, 'images')
        self.test_labels_path = os.path.join(self.test_path, 'labels')

        os.mkdir(self.train_images_path)
        os.mkdir(self.train_labels_path)
        os.mkdir(self.val_images_path)
        os.mkdir(self.val_labels_path)
        os.mkdir(self.test_images_path)
        os.mkdir(self.test_labels_path)

        # Create the images and labels folders for the whole dataset
        os.mkdir(self.images_path_out)
        os.mkdir(self.labels_path)




    def split_data(self):
        # Create the train, validation and test sets from the available images
        samples = [sample.split('.')[0] for sample in self.data['filename'].unique()]
        train_imgs, val_imgs = train_test_split(samples, test_size=self.val_split, random_state=42)
        train_imgs, test_imgs = train_test_split(train_imgs, test_size=0.2, random_state=42)
        return train_imgs, val_imgs, test_imgs
    
    def check_existance_img_and_label(self, img_name, type):
        # Check if the image and the label 
        img_name = img_name + '.tif'
        img_path = os.path.join(self.images_path, img_name)
        # Check the label exist in the dataframe self.data
        annotations_count = self.data[self.data['filename'] == img_name].shape[0]
        #label_path = os.path.join(self.annotations, img_name.split('.')[0] + '.xml')
        if os.path.exists(img_path) and annotations_count>10: #os.path.exists(label_path):
            return True
        else:
            print(f'Image {img_path} or annotation not found')
            print(f'Removing {img_name} from the {type} set')
            return False
    
    def preprocess(self):
        # TODO 
        return

    def create_patches(self):
        # TODO 
        # Create patches of the images and the labels and save them in the train, val and test folders
        # create_patches(img_filename, images_path, df, destination, class_to_num, slice_size=512, overlapping=0.2)
        self.patches_info = pd.DataFrame(columns=['filename',  'xmin', 'ymin', 'xmax', 'ymax', 'slice','path'])
        for img in self.train_imgs:
            img_name = img + '.tif'
            #img_path = os.path.join(self.images_path, img_name)
            try:
                #create_patches(img_name, self.images_path, self.data, self.train_path, self.class_to_num, self.slice_size, self.overlapping)
                self.patches_info = pd.concat([create_patches(img_name, self.images_path, self.data, self.train_path, self.class_to_num, self.slice_size, self.overlapping), self.patches_info], ignore_index=True)
            except:
                print(f'Error in creating patches for training image {img_name}')
                continue
        for img in self.val_imgs:
            img_name = img + '.tif'
            try:
                self.patches_info = pd.concat([create_patches(img_name, self.images_path, self.data, self.val_path, self.class_to_num, self.slice_size, self.overlapping), self.patches_info], ignore_index=True)
                #create_patches(img_name, self.images_path, self.data, self.val_path, self.class_to_num, self.slice_size, self.overlapping)
            except:
                print(f'Error in creating patches for validation image {img_name}')
                continue

        for img in self.test_imgs:
            img_name = img + '.tif'
            try:
                self.patches_info = pd.concat([create_patches(img_name, self.images_path, self.data, self.test_path, self.class_to_num, self.slice_size, self.overlapping), self.patches_info], ignore_index=True)
                #create_patches(img_name, self.images_path, self.data, self.test_path, self.class_to_num, self.slice_size, self.overlapping)
            except:
                print(f'Error in creating patches for test image {img_name}')
                continue
        
        #save patches info in a csv file
        self.patches_info.to_csv(os.path.join(self.destination, 'patches_info.csv'), index=False)
        
        print(self.patches_info)
        print('Patches created')
    
    def convert_annotations(self):
        # Create the YOLO format annotations for the train, val and test sets of the original images
        # Creating custom annotations for YOLO format
        print("\nCreating custom annotations for YOLO format...\n")
        print("FORMAT: class_num x_center y_center width height")
        self.data = yolo_format(self.data, self.class_to_num)
        #print(self.data)

    
    def create_yaml(self):
         #create data.yaml
        yaml_file = os.path.join(self.destination, 'data.yaml')

        with open(yaml_file, 'w') as wobj:
            wobj.write('train: ' + os.path.join(self.train_path, 'images'))
            wobj.write("\n")
            wobj.write('val: ' + os.path.join(self.val_path, 'images'))
            wobj.write("\n")
            wobj.write('test: ' + os.path.join(self.test_path, 'images'))
            wobj.write("\n")
            wobj.write('nc: ' + str(len(self.data['class'].unique())))
            wobj.write("\n")
            wobj.write('names: ' + str(list(self.data['class'].unique())))
            wobj.write("\n")

        wobj.close()
    
    def populate_train_val_test(self):
        print(self.data)
        
        if self.slicing:
            #TODO Create patches
            print('Creating patches...')
            self.create_patches()
            
        else:
            # Create YOLO format annotations
            
            self.convert_annotations()
            # Copy the images and labels in the train, val and test folders
            for img in self.train_imgs:
                img_name = img + '.tif'
                img_path = os.path.join(self.images_path, img_name)
                #label_path = os.path.join(self.annotations, img_name.split('.')[0] + '.xml')
                shutil.copy(img_path, self.train_images_path)
                # shutil.copy(label_path, self.train_labels_path)
                # Create label file for the image
                img_data = self.data[self.data['filename'] == img_name]
                img_data = img_data[['class_num', 'x_center', 'y_center', 'bbox_width', 'bbox_height']]
                img_data.to_csv(os.path.join(self.train_labels_path, img_name.split('.')[0] + '.txt'), header=None, index=None, sep=' ')

            
            for img in self.val_imgs:
                img_name = img + '.tif'
                img_path = os.path.join(self.images_path, img_name)
                label_path = os.path.join(self.annotations, img_name.split('.')[0] + '.xml')
                shutil.copy(img_path, self.val_images_path)
                # Create label file for the image
                img_data = self.data[self.data['filename'] == img_name]
                img_data = img_data[['class_num', 'x_center', 'y_center', 'bbox_width', 'bbox_height']]
                img_data.to_csv(os.path.join(self.val_labels_path, img_name.split('.')[0] + '.txt'), header=None, index=None, sep=' ')
            
            for img in self.test_imgs:
                img_name = img + '.tif'
                img_path = os.path.join(self.images_path, img_name)
                label_path = os.path.join(self.annotations, img_name.split('.')[0] + '.xml')
                shutil.copy(img_path, self.test_images_path)
                # Create label file for the image
                img_data = self.data[self.data['filename'] == img_name]
                img_data = img_data[['class_num', 'x_center', 'y_center', 'bbox_width', 'bbox_height']]
                img_data.to_csv(os.path.join(self.test_labels_path, img_name.split('.')[0] + '.txt'), header=None, index=None, sep=' ')
        
        # Create data.yaml
        self.create_yaml()




if __name__ == "__main__":
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Converts xml annotations to YOLO format and creates train, test and validation sets')
    
    # Add the arguments
    my_parser.add_argument('--path',
                           type=str,
                           help='the path to the folder containing the xml files',
                           default='./dataset_demo')
    my_parser.add_argument('--destination',
                           type=str,
                           help='the path to the destination folder',
                           default='./processed_data')
    my_parser.add_argument('--slice-size',
                           type=int,
                           help='the size of the patches',
                           default=512)
    my_parser.add_argument('--overlapping',
                           type=int,
                           help='the overlapping percentage between patches',
                           default=0.2)
    my_parser.add_argument('--slicing',
                           action='store_true',
                           help='True if you want to slice the images, False otherwise',
                           default=False)
    my_parser.add_argument('--val-split',
                           type=int,
                           help='Validation split to generate',
                           default=0.2)
    my_parser.add_argument('--train-imgs', nargs='+', type=str, help='Specify here the tiles you want to use for training e.g.Format: P9 P20 N10', default=[])
    my_parser.add_argument('--val-imgs', nargs='+', type=str, help='Specify here the tiles you want to use for validation e.g.Format: P9 P20 N10', default=[])
    my_parser.add_argument('--test-imgs', nargs='+', type=str, help='Specify here the tiles you want to use for inference e.g.Format: P9 P20 N10', default=[])
       
    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    print(args)
    path = args.path
    destination = args.destination
    slice_size = args.slice_size
    overlapping = args.overlapping
    slicing = args.slicing
    val_split = args.val_split
    train_imgs = args.train_imgs
    val_imgs = args.val_imgs
    test_imgs = args.test_imgs
    
    # Create the dataset object
    dataset = DataProcessor(path, destination, slice_size, overlapping, slicing, val_split, train_imgs, val_imgs, test_imgs)

    dataset.populate_train_val_test()


