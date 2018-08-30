from PIL import Image
import os
import os.path
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, root_path, data_folder='train', name_list='ucfTrainTestlist', version=1, transform=None,
                 num_frames=16, modality='RGB', channel=3, size=160):
        self.root_path = root_path
        self.num_frames = num_frames
        self.data_folder = data_folder
        self.size = size
        self.channel = channel
        self.modality = modality

        self.split_file = os.path.join(self.root_path, name_list,
                                       str(data_folder) + 'list0' + str(version) + '.txt')
        self.label_file = os.path.join(self.root_path, name_list, 'classInd.txt')
        self.label_dict = self.get_labels()

        self.video_dict = self.get_video_list()

        self.version = version
        self.transform = transform

    def get_video_list(self):
        res = []
        with open(self.split_file) as fin:
            for line in list(fin):
                line = line.replace("\n", "")
                split = line.split(" ")
                # get number frames of each video
                video_path = split[0].split('.')[0]
                frames_path = os.path.join(self.root_path, self.data_folder, video_path)
                allfiles = glob.glob(frames_path + '/*.jpg')
                # remove video which has < 16 image frames
                if len(allfiles) >= self.num_frames:
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
        seed = np.random.random_integers(0, len(images) - self.num_frames)  # random sampling
        clip = list()
        for i in range(self.num_frames):
            img = Image.open(images[i + seed])
            # img = img.convert('RGB')
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


if __name__ == "__main__":
    from transforms import *
    import video_transforms

    num_frames = 20
    channel = 3
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_frames=num_frames, channel=channel)
    transform = Compose([
        Resize((182, 242)),
        CenterCrop(160),
        ToTensor(),
        normalize,
    ])


    normalize = video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transformations = video_transforms.Compose([
        video_transforms.Resize((182, 242)),
        video_transforms.CenterCrop(160),
        video_transforms.ToTensor(),
        normalize
    ])

    train_dataset = MyDataset(
        root_path='/Users/naviocean/data/UCF101/',
        data_folder="validation",
        name_list='ucfTrainTestlist',
        version="1",
        transform=transform,
        num_frames=num_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=15,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        if i == 1:
            break
    print('xxxx')
    train_dataset = MyDataset(
        root_path='/Users/naviocean/data/UCF101/',
        data_folder="validation",
        name_list='ucfTrainTestlist',
        version="1",
        transform=val_transformations,
        num_frames=16
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=15,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        if i == 1:
            break

