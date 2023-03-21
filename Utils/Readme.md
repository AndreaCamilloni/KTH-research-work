# Utils

## Data Processor class

Class used to pre-process the data and create train, val and test split.
How to use it (EXAMPLE):

        python dataProcessor.py --train-imgs N10_1_2 N10_2_1 N10_2_2 P20_1_3 P20_1_4 P20_2_3 P20_6_1 P20_7_1 P20_3_1 P20_3_2 P20_2_4 P7_HE_Default_Extended_4_2 P9_3_1 P9_2_1 P14_HE_Default_Extended_1_1 P14_HE_Default_Extended_1_2 P14_HE_Default_Extended_2_1 P14_HE_Default_Extended_2_2 P19_2_1 P19_3_2 P11_1_1 P11_1_2 HE_T12193_90_Default_Extended_1_1 HE_T3482_84_Default_Extended_1_1 HE_T1087_84_Default_Extended_1_3 P25_3_1 P28_8_5 P13_1_2 P13_2_1  --test-imgs N10_5_2 P20_4_1 P7_HE_Default_Extended_3_2 P7_HE_Default_Extended_4_1 P13_1_1 P14_HE_Default_Extended_3_1 P14_HE_Default_Extended_3_2 HE_T12193_90_Default_Extended_1_2 HE_T3482_84_Default_Extended_2_1 P11_2_1 P11_2_2 P19_2_2 P19_3_1 P25_8_2 P28_7_5  --val-imgs HE_T1087_84_Default_Extended_1_2 P14_HE_Default_Extended_2_3 P20_5_1 P14_HE_Default_Extended_3_3 P11_3_1 P11_3_2 P9_2_1 --slicing

## Convert2XML

Class used to restore XML annotations from YOLO predicted labels.

        python convert2xml.py 