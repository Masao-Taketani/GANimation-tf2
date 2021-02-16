import os
from glob import glob


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


def get_data(data_dir):
    img_list = get_files(data_dir, True)
    cond_list = get_files(data_dir, False)
    img_ids = read_ids(img_list)
    cond_ids = read_ids(cond_list)

    data = get_intersected_data(img_ids, cond_ids)
    print(data)


if __name__ == "__main__":
    data_dir = "../dataset"
    get_data(data_dir)