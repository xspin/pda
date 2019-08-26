import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))

def load_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_image_path(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith('jpg') or x.name.endswith('png')]

def get_label_image_path(directory, labels=None):
    assert os.path.exists(directory), 'Directory `%s` does not exist.' % directory
    if labels is None or (type(labels) is str and labels.lower() == 'all'):
        labels = os.listdir(directory)
    img_paths = []
    img_labels = []
    for lb in labels:
        label_dir = os.path.join(directory,lb)
        assert os.path.exists(label_dir), 'Not found label `%s`.' % lb
        paths = get_image_path(label_dir)
        img_paths.extend(paths)
        img_labels.extend([lb]*len(paths))
    return img_paths, img_labels

class MyDataset(Dataset):
    def __init__(self, path, labels=None, transform=None, label2index=None):
        self.image_files, self.labels = get_label_image_path(path, labels)
        labels_set = sorted(list(set(self.labels)))
        self.label2index = {lb:i for i,lb in enumerate(labels_set)} if label2index is None else label2index 
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()]) 

    def __getitem__(self, index):
        return self.transform(load_image(self.image_files[index])),\
            self.label2index[self.labels[index]]

    def __len__(self):
        return len(self.image_files)

if __name__ == "__main__":
    amazon_dir = 'datasets/office31/amazon/images'
    # fl = get_image_path(amazon_dir)
    fl = get_label_image_path(amazon_dir, ['bike'])
    print(fl[0])
