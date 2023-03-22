# YOLOv5 - Custom implementation

## How to train:

        python train.py --hyp data/hyps/hyp.scratch.yaml --cfg models/yolov5x.yaml --data data/new_data_processed_1.yaml --batch-size 4 --epochs 500 --img-size 512 --project runs/vanilla_exp/train --name Experiment_New_Data_v5x_500epochs_aug --device 0 --patience 100

Make sure that the dataset.yaml file contains the 1) paths to train / val / test image directories (or *.txt files with image paths) and 2) a class names dictionary, and 3) that is under the folder data/

        Experiment--yolov5_custom--classify
            |              |-------data
            |              |-------modoels
            |              |-------utils
            |              |------- ...
            |-------dataset--------train
                       |-----------val
                       |-----------test
            