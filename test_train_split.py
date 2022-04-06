import random
from sklearn.model_selection import train_test_split
import os
from distutils.dir_util import copy_tree #copies entire dir structure


def segregate_emotions(path: os.path) -> None:
    label = []
    for folder_name in sorted(os.listdir(path)):
        if folder_name.startswith('.') is False:
            identifiers = folder_name.split('_')
            if identifiers[2] == 'happy' or identifiers[2] == 'sad':
                label.append(identifiers[2])
                copy_tree(os.path.join(path, folder_name), os.path.join(path, '../dataset', folder_name))

    x_train, x_test, y_train, y_test = train_test_split(
        os.listdir(os.path.join(path, '../dataset')), label, test_size=0.2, random_state=42)


def segregate_into_train_test(path: os.path) -> None:
    os.mkdir(os.path.join(path, '../train'))
    os.mkdir(os.path.join(path, '../test'))

    unique_actors = set()

    for folder in os.listdir(path):
        if folder.startswith('.') is False:
            identifiers = folder.split('_')
            unique_actors.add(identifiers[0])

    print(unique_actors)

    for actor in unique_actors:
        prob = random.random()
        for folder in os.listdir(path):
            if folder.startswith(actor):
                if prob <= 0.8:
                    copy_tree(os.path.join(path, folder), os.path.join(path, '../train', folder))
                else:
                    copy_tree(os.path.join(path, folder), os.path.join(path, '../test', folder))


def convert_train_to_txt(path):
    files = os.listdir(path)
    with open('/home/sg/AV/train.txt', 'w') as txtfile:
        for file in files:
            txtfile.write(file)
            txtfile.write('\n')


def convert_test_to_txt(path):
    files = os.listdir(path)
    with open('/home/sg/AV/val.txt', 'w') as txtfile:
        for file in files:
            txtfile.write(file)
            txtfile.write('\n')
