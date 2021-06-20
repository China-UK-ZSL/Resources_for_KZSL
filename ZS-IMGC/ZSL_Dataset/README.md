# ZS-IMGC dataset AwA and ImNet-A/O

## Dataset Preparation and Illustrations


### ImageNet (ImNet-A, ImNet-O)
Download image features and class splits of ImageNet classes from [here](https://drive.google.com/drive/folders/1An6nLXRRvlKSCbJoKKlqTNDvgN7PyvvW) and put them to the folder `ImageNet/`.

1. The class split file `seen.txt` and `unseen.txt` separately list the WordNet ids of seen and unseen classes in ImNet-A/O.

2. The downloaded image feature folder `Res101_Features` contains three sub-folders:
    - ILSVRC2012_train: training set features (for all seen classes)
    - ILSVRC2012_val: testing set features for seen classes
    - ILSVRC2011: testing set features for unseen classes

&ensp;&ensp;&ensp;&ensp; In each sub-folder, each `.mat` file corresponds to one class and is named by the index of this class in `split.mat`.

3. `split.mat` includes the following fields:

    - allwnids: the WordNet ids of all ImageNet classes
    - allwords: the class names
    - seen: all seen classes in ImageNet (i.e., the ImageNet 2012 1K subset)
    - hops2: the classes that are within 2-hops of the seen classes according to the WordNet hierarchy
    - hops3: the classes that are within 3-hops of the seen classes according to the WordNet hierarchy
    - rest: all the rest classes in ImageNet 2011 21K after removing the 2/3-hops classes
    - no_w2v_index: the classes with no pre-trained word vectors

### AwA
Download public image features and dataset split for [AwA](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `AwA/`.

1. The class splits consist of:
    - allclasses.txt: list of names of all classes in the dataset
    - trainvalclasses.txt: seen classes
    - testclasses.txt: unseen classes


2. `att_splits.mat` includes the following fields:
    - att: columns correpond to class attribute vectors normalized to have unit l2 norm, following the classes order in allclasses.txt
    - original_att: the original class attribute vectors without normalization
    - trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
    - test_seen_loc: instances indexes of test set features for seen classes
    - test_unseen_loc: instances indexes of test set features for unseen classes


3. `resNet101.mat` includes the following fields:
    - features: columns correspond to all image instances in the dataset
    - labels: the labels of all images in features, and the label number of a class is its row number in allclasses.txt
    - image_files: original image sources





