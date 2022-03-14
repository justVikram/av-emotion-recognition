import shutil

from sklearn.model_selection import train_test_split
import os
from distutils.dir_util import copy_tree


def test_train_split(path: os.path) -> None:
    label = []
    for folder_name in sorted(os.listdir(path)):
        if folder_name.startswith('.') is False:
            identifiers = folder_name.split('_')
            if identifiers[2] == 'happy' or identifiers[2] == 'sad' \
                    or identifiers[2] == 'angry' or identifiers[2] == 'neutral':
                label.append(identifiers[2])
                copy_tree(os.path.join(path, folder_name), os.path.join(path, '../dataset', folder_name))

    x_train, x_test, y_train, y_test = train_test_split(
        os.listdir(os.path.join(path, '../dataset')), label, test_size=0.2, random_state=42)


if __name__ == '__main__':
    test_train_split('/Users/avikram/Projects/av-emotion-recognition/expt')
