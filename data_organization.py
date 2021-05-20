import os
import shutil
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from simplest_cb import simplest_cb

# input folder
data_root = "../data/Microfacies/Henry"
# readme update ('a') or new ('w')
readme_w = 'w'

# number of samples per class (except the NoClass ones)
n_test_samples = 5

today = datetime.today().strftime('%Y-%m-%d')

os.path.isdir(data_root)

os.listdir(data_root)


def add_dir_name(dir_in):
    """Adds the directory name into the name of the file
    and moves the file up one level (to the same level as dir_in)

    Args:
        dir_in (string/os.path): directory name
    """
    fnames = os.listdir(dir_in)

    for fname in tqdm(fnames):
        # change space to _
        new_name = fname.replace(" ", "_")
        # add directory name:
        new_name = 'sample_' + os.path.basename(dir_in) + "_" + new_name
        shutil.move(os.path.join(dir_in, fname),
                    os.path.join(os.path.dirname(dir_in), new_name))


# execute function above for all directories
# # start with samples without class
# add_dir_name(os.path.join(data_root, '1'))
# add_dir_name(os.path.join(data_root, '82'))

# do the same for the remaining
for root, dirs, files in os.walk(data_root):
    if len(dirs) > 0:
        if root != data_root:
            for current_dir in dirs:
                add_dir_name(os.path.join(root, current_dir))

for root, dirs, files in os.walk(data_root):

    for f in files:
        # remove xml files and cross polarized images:
        if f.endswith('xml') or f.endswith(".ini") or "XP" in f:
            #print(os.path.join(root, f))
            os.remove(os.path.join(root, f))

        # compute color balanced image:
        else:
            img = cv2.imread(os.path.join(root, f))
            img = simplest_cb(img, 1)
            cv2.imwrite(os.path.join(root, f), img)

    # remove empty directories
    if len(dirs) == 0:
        shutil.rmtree(root)

# add NoClass folder:
# os.mkdir(os.path.join(data_root, "NoClass"))
# for fname in os.listdir(data_root):
#     if os.path.isfile(os.path.join(data_root, fname)):
#         shutil.move(os.path.join(data_root, fname),
#                     os.path.join(data_root, "NoClass", fname))


# count number of samples:
for img_class in os.listdir(data_root):
    print(img_class, len(os.listdir(os.path.join(data_root, img_class))))

# move samples into train folder:
img_classes = os.listdir(data_root)
os.mkdir(os.path.join(data_root, 'train'))
train_dir = os.path.join(data_root, "train")
for img_class in img_classes:
    shutil.move(os.path.join(data_root, img_class),
                os.path.join(train_dir, img_class))

# create test folder:
test_dir = os.path.join(os.path.join(data_root, 'test'))
os.mkdir(test_dir)

# set seed
np.random.seed(123456789)
for img_class in tqdm(img_classes):
    if img_class != 'NoClass':
        os.mkdir(os.path.join(test_dir, img_class))
        all_train = os.listdir(os.path.join(train_dir, img_class))
        test_samples = np.random.choice(
            all_train, size=n_test_samples, replace=False)

        for test_sample in test_samples:
            shutil.move(os.path.join(train_dir, img_class, test_sample),
                        os.path.join(test_dir, img_class, test_sample))


# count number of samples in different folders:
print("class, train, test")
for img_class in img_classes:
    if img_class != 'NoClass':
        print(img_class,
              len(os.listdir(os.path.join(train_dir, img_class))),
              len(os.listdir(os.path.join(test_dir, img_class))), sep=",")

# update README.md file:
with open(os.path.join(data_root, 'Readme.md'), readme_w) as f:
    f.write(
        f"Folders created with `{os.path.basename(__file__)}` on {today}.\n")
    f.write('The images were color balanced with the `simplest_cb` function in `simplest_cb.py`\n')
    f.write('| Class         | Train | Test |\n')
    f.write('| ------------- | ----- | ---- |\n')
    for img_class in img_classes:
        if img_class != 'NoClass':
            len_train = len(os.listdir(os.path.join(train_dir, img_class)))
            len_test = len(os.listdir(os.path.join(test_dir, img_class)))
            f.write(f'|{img_class} | {len_train} | {len_test} |\n')
