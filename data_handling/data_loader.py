import os
import random
from glob import glob
import copy

from tqdm import tqdm
import numpy as np


def get_files(data_dir, is_img):
    if is_img:
        return glob(os.path.join(data_dir, "*.jpg"))
    else:
        return glob(os.path.join(data_dir, "*.csv"))


def read_ids(data_list):
    ids_list = []
    for data in data_list:
        ids_list.append(data[:-4])
    
    return ids_list


def get_intersected_data(img_ids, cond_ids):
    return set(img_ids).intersection(set(cond_ids))


def convert_labels(csv_path):
    # From csv to AU values
    cond = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if len(cond.shape) > 1:
        conf_max_idx = np.argmax(cond, axis=0)[1]
        cond = cond[conf_max_idx]
    # Normalize values
    cond = cond[2:19] / 5.0
    assert cond.shape == (17,), "{} has a different shape, which is {}".format(csv_path,
                                                                               cond.shape)
    cond = list(cond)
    
    return cond


def get_random_fin_conds(ini_lbl_list):
    fin_lbl_list = np.copy(ini_lbl_list)
    np.random.shuffle(fin_lbl_list)
    # add uniform noise
    fin_lbl_list += np.random.uniform(-0.1, 0.1, fin_lbl_list.shape)

    return fin_lbl_list.tolist()
    

def get_data(data_dir):
    ipath_list = []
    ini_lbl_list = []

    img_list = get_files(data_dir, True)
    cond_list = get_files(data_dir, False)
    img_ids = read_ids(img_list)
    cond_ids = read_ids(cond_list)

    data_ids = get_intersected_data(img_ids, cond_ids)
    print("number of data:", len(data_ids))
    random.seed(1234)
    data_ids = list(data_ids)
    random.shuffle(data_ids)
    train_ids = data_ids[:200000]

    for ipath in tqdm(train_ids):
        ipath_list.append(ipath + ".jpg")
        ini_cond = convert_labels(ipath + ".csv")
        ini_lbl_list.append(ini_cond)
    #fin_lbl_list = get_random_fin_conds(ini_lbl_list)

    return ipath_list, ini_lbl_list


if __name__ == "__main__":
    data_dir = "../dataset/cropped/"
    ipath_list, ini_lbl_list = get_data(data_dir)
    print("ipath shape:", len(ipath_list))
    print("ini lbl shape:", len(ini_lbl_list), len(ini_lbl_list[0]))
    #print("fin lbl shape:", len(fin_lbl_list), len(fin_lbl_list[0]))