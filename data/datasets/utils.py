import numpy as np


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, "rt").read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(" ")[0] for s in data_file_list]
        file_label = [int(s.split(" ")[1]) for s in data_file_list]

    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [
            k for k, v in enumerate(train_color_label) if v == unique_label_color[i]
        ]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [
            k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]
        ]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


def GenCamIdx(gall_img, gall_label, mode):
    if mode == "indoor":
        camIdx = [1, 2]
    else:
        camIdx = [1, 2, 4, 5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))

    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [
                k
                for k, v in enumerate(gall_label)
                if v == unique_label[i] and gall_cam[k] == camIdx[j]
            ]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
        # cam_id = 2
        gall_cam.append(cam_id)

    return np.array(gall_cam)

