# YOLOv5 - Custom implementation

## How to train:

        python train.py --weights '' --cfg models/yolov5x.yaml --data data/new_data_processed_1.yaml --batch-size 16 --epochs 100 --img-size 512 --project runs/vanilla_exp/train --name Experiment_New_Data_v5x --device 0 --patience 100

Make sure that the dataset.yaml file contains the 1) paths to train / val / test image directories (or *.txt files with image paths) and 2) a class names dictionary, and 3) that is under the folder yolov5_custom/data/

        Experiment--yolov5_custom--classify
            |              |-------data
            |              |-------modoels
            |              |-------utils
            |              |------- ...
            |-------dataset--------train
                       |-----------val
                       |-----------test

The file dataset.yaml should look like the following:

        train: ./processed_data_8\train\images
        val: ./processed_data_8\val\images
        test: ./processed_data_8\test\images
        nc: 4
        names: ['epithelial', 'lymphocyte', 'fibroblast and endothelial', 'inflammatory']
           

## Run Validation:

        python val.py --task val --weights runs\vanilla_exp\train\Experiment_New_Data_v5x\weights\best.pt --data data/new_data_processed_1.yaml --batch-size 16 --img-size 512 --project runs/vanilla_exp/val --name Experiment_New_Data_v5x --device 0 --save-txt

## Run inference:

        python val.py --task test --weights runs\vanilla_exp\train\Experiment_New_Data_v5x\weights\best.pt --data data/new_data_processed_1.yaml --batch-size 16 --img-size 512 --project runs/vanilla_exp/test --name Experiment_New_Data_v5x --device 0 --save-txt
        
## AUGMENTATION

### HSV AUGMENTATION 
<p align="center">

<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/HSV-SAT.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/HSV-VALUE.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/HSV-HUE.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/HSV.png>

</p>

### Image space augmentation
<p align="center">
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/SCALE.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/SCALE-TRANSLATION.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/ROTATION.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/SHEAR.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/MIXUP.png>
<img src= https://github.com/AndreaCamilloni/MScThesis-KTHWork/blob/1a0bda42fb62c5a79e91097bc706090dfa061f1a/yolov5_custom/figures/AUG-BATCH/COPYPASTE.png>
 </p>