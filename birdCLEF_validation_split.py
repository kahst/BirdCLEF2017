import os
from shutil import copyfile
from sklearn.utils import shuffle

######################################################

#Specify the source folder containing subfolders named after genus, species and class id
#Use birdCLEF_sort_data.py in order to sort wav files accordingly
train_path = 'dataset/train/src/'

#Specify target folder for validation split
test_path = 'dataset/val/src/'

######################################################

#get classes from subfolders
classes = [c for c in sorted(os.listdir(train_path))]

#get files for classes
for c in classes:

    #shuffle files
    files = shuffle([train_path + c + "/" + f for f in os.listdir(train_path + c)], random_state=1337)

    #choose amount of files for validation split from each class
    #we want at least 1 sample per class (2 if sample count os between 12 and 20)
    #we take 10% of the samples if sample count > 20
    if len(files) <= 12:
        num_test_files = 1
    elif len(files) > 12 and len(files) <= 20:
        num_test_files = 2
    else:
        num_test_files = int(len(files) * 0.1)

    test_files = files[:num_test_files]
    print c, len(files), len(test_files)

    #copy test files for validation to target folder
    for tf in test_files:

        #copy test file
        new_path = tf.replace(train_path, test_path).rsplit("/", 1)[0]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        copyfile(tf, tf.replace(train_path, test_path))

        #remove test file from train
        #Note: You might want to test the script first before deleting any files :)
        os.remove(tf)
        
    #break
