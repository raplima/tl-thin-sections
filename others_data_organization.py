import os
import shutil
import zipfile
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.compute_mean_std import compute_mean_std


def main():
    data_root = "../data/Others/"
    # readme update ('a') or new ('w')
    readme_w = 'w'

    # datasets
    dsets = ['train', 'test']

    today = datetime.today().strftime('%Y-%m-%d')

    # remove useless files:
    for root, dirs, files in tqdm(os.walk(data_root), desc='Deleting unused files'):
        for f in files:
            if os.path.splitext(f)[1] == '.ini':
                os.remove(os.path.join(root, f))
    ##############################################################################
    # 1
    # Histological images for MSI vs. MSS classification in gastrointestinal cancer, FFPE samples
    # this dataset is already organized into train and test sets
    # we just have to unzip the files and move them to the appropriate location
    ##############################################################################
    img_classes = ['MSIMUT', 'MSS']
    dir_dset_root = os.path.join(data_root, 'Histological_images_MSI_vs_MSS')
    train_dir = os.path.join(dir_dset_root, 'train')
    test_dir = os.path.join(dir_dset_root, 'test')

    # create train and test folders:
    for dset in [train_dir, test_dir]:
        os.mkdir(dset)

    # extract files to appropriate folder
    for root, dirs, files in tqdm(os.walk(dir_dset_root), desc='MSI vs MSS files'):
        for f in files:
            if os.path.splitext(f)[1] == '.zip':
                for dset in dsets:
                    if dset in f.lower():
                        with zipfile.ZipFile(os.path.join(root, f), 'r') as zip_ref:
                            zip_ref.extractall(os.path.join(root, dset))

    # compute channel mean and std
    dset_mean, dset_std = compute_mean_std(train_dir, n_channels=3)

    # update README.md file:
    with open(os.path.join(dir_dset_root, 'Readme.md'), readme_w) as f:
        f.write(f"Folders created with `{os.path.basename(__file__)}` ")
        f.write(f"on {today}.\n")
        f.write('Examples for colorectal cancer from ')
        f.write('[Zenodo](https://zenodo.org/record/2530835#.YJ6LVKj0lPa). ')
        f.write('Study published in ')
        f.write('[Nature](https://www.nature.com/articles/s41591-019-0462-y). \n')
        f.write('| Class         | Train | Test |\n')
        f.write('| ------------- | ----- | ---- |\n')
        for img_class in img_classes:
            if img_class != 'NoClass':
                len_train = len(os.listdir(
                    os.path.join(train_dir, img_class)))
                len_test = len(os.listdir(os.path.join(test_dir, img_class)))
                f.write(f'|{img_class} | {len_train} | {len_test} |\n')
        f.write(f'Train mean: {dset_mean}. \n')
        f.write(f'Train std: {dset_std}')

    ##############################################################################
    # 2
    # HAM10000
    # The labels for this dataset are stored in a different file.
    # We use pandas to split the data and to move the files to
    # the appropriate location
    ##############################################################################
    dir_dset_root = os.path.join(data_root, 'HAM10000')
    dir_all_images = os.path.join(dir_dset_root, 'all_images')
    train_dir = os.path.join(dir_dset_root, 'train')
    test_dir = os.path.join(dir_dset_root, 'test')

    # start by extracting the HAM10000 file:
    with zipfile.ZipFile(os.path.join(data_root, 'HAM10000.zip'), 'r') as zip_ref:
        zip_ref.extractall(dir_dset_root)

    # extract all image files into the all_images folder:
    for f in ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']:
        with zipfile.ZipFile(os.path.join(dir_dset_root, f), 'r') as zip_ref:
            zip_ref.extractall(dir_all_images)

    # read in the metadata file:
    df = pd.read_csv(os.path.join(dir_dset_root, 'HAM10000_metadata'))

    # create class folders:
    img_classes = df['dx'].unique()
    for dset in [train_dir, test_dir]:
        # create train and test folders:
        os.mkdir(dset)
        for cl in img_classes:
            # create class folders:
            os.mkdir(os.path.join(dset, cl))

    # split into train and test
    train_df, test_df = train_test_split(df,
                                         test_size=0.2,
                                         random_state=42,
                                         shuffle=True)
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['dset_dir'] = train_dir
    test_df['dset_dir'] = test_dir

    def move_imgs(row):
        """Move files according to the label
        !!!! uses global variables
        Args:
            row (pandas row): row of the pandas dataframe

        """
        # move files according to the label
        shutil.move(src=os.path.join(dir_all_images, f"{row['image_id']}.jpg"),
                    dst=os.path.join(row['dset_dir'],
                                     row['dx'],
                                     f"{row['image_id']}.jpg"))
        return None

    train_df.apply(move_imgs, axis=1)
    test_df.apply(move_imgs, axis=1)

    # compute channel mean and std
    dset_mean, dset_std = compute_mean_std(train_dir, n_channels=3)

    # update README.md file:
    with open(os.path.join(dir_dset_root, 'Readme.md'), readme_w) as f:
        f.write(f"Folders created with `{os.path.basename(__file__)}` ")
        f.write(f"on {today}.\n")
        f.write('Examples for pigmented skin lesions from [HAM10000]')
        f.write(
            '(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). ')
        f.write('Study published in ')
        f.write('[Nature](https://www.nature.com/articles/sdata2018161). \n')
        f.write('| Class         | Train | Test |\n')
        f.write('| ------------- | ----- | ---- |\n')
        for img_class in img_classes:
            if img_class != 'NoClass':
                len_train = len(os.listdir(
                    os.path.join(train_dir, img_class)))
                len_test = len(os.listdir(os.path.join(test_dir, img_class)))
                f.write(f'|{img_class} | {len_train} | {len_test} |\n')
        f.write(f'Train mean: {dset_mean}. \n')
        f.write(f'Train std: {dset_std}')

    ##############################################################################
    # 3
    # rawfoot
    # this data is already organized
    # update Readme only
    ##############################################################################
    dir_dset_root = os.path.join(data_root, 'rawfoot-tiles')
    train_dir = os.path.join(dir_dset_root, 'train')
    test_dir = os.path.join(dir_dset_root, 'test')

    img_classes = [d for d in os.listdir(train_dir) if '.ini' not in d]

    # compute channel mean and std
    dset_mean, dset_std = compute_mean_std(train_dir, n_channels=3)

    # update README.md file:
    with open(os.path.join(dir_dset_root, 'Readme.md'), readme_w) as f:
        f.write("Folders organized by authors.\n")
        f.write('Examples for raw food texture [rawfoot]')
        f.write('(http://projects.ivl.disco.unimib.it/minisites/rawfoot/). ')
        f.write(
            'Study published in [Journal of the Optical Society of America]')
        f.write(
            '(https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-33-1-17). \n')
        f.write('| Class         | Train | Test |\n')
        f.write('| ------------- | ----- | ---- |\n')
        for img_class in img_classes:
            if img_class != 'NoClass':
                len_train = len(os.listdir(
                    os.path.join(train_dir, img_class)))
                len_test = len(os.listdir(os.path.join(test_dir, img_class)))
                f.write(f'|{img_class} | {len_train} | {len_test} |\n')
        f.write(f'Train mean: {dset_mean}. \n')
        f.write(f'Train std: {dset_std}')


if __name__ == "__main__":
    main()
