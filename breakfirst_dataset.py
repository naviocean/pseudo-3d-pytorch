from PIL import Image
import os
import os.path
import glob
import numpy as np
from torch.utils.data import Dataset


class BreakfastDataset(Dataset):
    def __init__(self, root_path, data_folder, version=1, transform=None, num_frame=16, modality='RGB'):
        self.root_path = root_path
        self.data_folder = data_folder
        self.split_file = os.path.join(self.root_path,
                                       'breakfastTrainTestList/' + str(data_folder) + 'list0' + str(version) + '.txt')
        self.label_file = os.path.join(self.root_path, 'breakfastTrainTestList/classInd.txt')
        self.label_dict = self.get_labels()

        self.video_dict = self.get_video_list()

        self.version = version
        self.transform = transform
        self.num_frame = num_frame

    def get_video_list(self):
        res = []
        with open(self.split_file) as fin:
            for line in list(fin):
                line = line.replace("\n", "")
                split = line.split(" ")
                res.append(split[0])
        return res

    # Get all labels from classInd.txt
    def get_labels(self):
        label_dict = {}
        with open(self.label_file) as fin:
            for row in list(fin):
                row = row.replace("\n", "").split(" ")
                # -1 because the index of array is start from 0
                label_dict[row[1]] = int(row[0]) - 1
        return label_dict

    # Get all frame images of video
    def get_all_images(self, dir, file_ext="jpg", sort_files=True):
        allfiles = glob.glob(dir + '/*.' + file_ext)
        if sort_files and len(allfiles) > 0:
            allfiles = sorted(allfiles)
        return allfiles

    def get_video_tensor(self, dir):
        images = self.get_all_images(dir)
        seed = np.random.random_integers(0, len(images) - self.num_frame)  # random sampling
        clip = list()
        for i in range(self.num_frame):
            img = Image.open(images[i + seed])
            clip.append(img)
        clip = self.transform(clip)
        return clip

    # stuff
    def __getitem__(self, index):
        video = self.video_dict[index]
        video_path = video.split('.')[0]

        frames_path = os.path.join(self.root_path, self.data_folder, video_path)
        clip = self.get_video_tensor(frames_path)
        # get label name from video path
        label_name = video_path.split('/')[0]
        label_index = self.label_dict[label_name];
        return (clip, label_index)

    def __len__(self):
        return len(self.video_dict)
