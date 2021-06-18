# ZS-IMGC dataset AwA and ImNet-A/O

## Dataset Preparation and Illustrations

### AwA
Download public image features and dataset split for [AwA](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `AwA/`.

`resNet101.mat` includes the following fields:
- features: columns correspond to image instances
- labels: label number of a class is its row number in allclasses.txt
- image_files: image sources


`att_splits.mat` includes the following fields:
- att: columns correpond to class attribute vectors normalized to have unit l2 norm, following the classes order in allclasses.txt
- original_att: the original class attribute vectors without normalization
- trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
- test_seen_loc: instances indexes of test set features for seen classes
- test_unseen_loc: instances indexes of test set features for unseen classes



### ImageNet (ImNet-A, ImNet-O)
- Download image features of ImageNet classes from [here](https://drive.google.com/drive/folders/1An6nLXRRvlKSCbJoKKlqTNDvgN7PyvvW) and put them to the folder `ImageNet/`.
- Class splits have been provided in the corresponding folders