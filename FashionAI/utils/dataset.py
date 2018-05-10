#-*-coding:utf-8-*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import csv
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path


AttrValues = {'skirt_length_labels':['Invisible','Short_Length','Knee_Length','Midi_Length','Ankle_Length','Floor_Length'],\
            'coat_length_labels':['Invisible','High_Waist_Length','Regular_Length','Long_Length','Micro_Length','Knee_Length','Midi_Length','Ankle&Floor_Length'],\
            'collar_design_labels':['Invisible','Shirt_Collar','Peter_Pan','Puritan_Collar','Rib_Collar'],\
            'lapel_design_labels':['Invisible','Notched','Collarless','Shawl_Collar','Plus_Size_Shawl'],\
            'neck_design_labels':['Invisible','Turtle_Neck','Ruffle_Semi-High Collar','Low_Turtle_Neck','Draped_Collar'],\
            'neckline_design_labels':['Invisible','Strapless_Neck','Deep_V_Neckline','Straight_Neck','V_Neckline','Square_Neckline','Off_Shoulder','Round_Neckline','Sweat_Heart_Neck','One_Shoulder_Neckline'],\
            'pant_length_labels':['Invisible','Short_Pant','Mid_Length','3/4_Length','Cropped_Pant','Full_Length'],\
            'sleeve_length_labels':['Invisible','Sleeveless','Cup Sleeves','Short Sleeves','Elbow Sleeves','3/4 Sleeves','Wrist Length','Long Sleeves','Extra Long Sleeves']}

# check if file is a image
# filename: Images/.../xxx.jpg
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

# class_to_idx: from 0 to 99
# classes: folder name
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# root: /media/tsq/Elements/阿里Fashion数据/base
# path = root+ImgPath(Images/coat_length_labels/xxx.jpg)
def make_dataset(root, csvPath, AttrKey, extensions):
    images = []
    with open(csvPath,'rb') as csv_file:
        fieldnames = ['ImgPath', 'AttrKey1', 'PreLabel']
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            ImgPath, AttrKey1, PreLabel = row
            if AttrKey1==AttrKey:
                if has_file_allowed_extension(ImgPath, extensions):
                    path = os.path.join(root, ImgPath)
                    index = PreLabel.find('y')
                    item = (path, index)
                    images.append(item)

    return images

class ALI_DatasetFolder(data.Dataset):
    def __init__(self, root, AttrKey, csvPath, loader, extensions, transform=None, target_transform=None):
        classes = AttrValues[AttrKey]
        class_to_idx = {name:index for index,name in enumerate(classes)}
        
        samples = make_dataset(root, csvPath, AttrKey, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ALI_ImageFolder(ALI_DatasetFolder):
    """
        read csv file, and generate Dataset
    """
    def __init__(self, root, AttrKey, csvName='label.csv', transform=None, target_transform=None, 
                loader=default_loader):
        csvPath = os.path.join(root,'Annotations',csvName)
        super(ALI_ImageFolder, self).__init__(root, AttrKey, csvPath, loader, 
                                          IMG_EXTENSIONS, transform=transform, 
                                          target_transform=target_transform)

        self.imgs = self.samples


def train_loader(root, AttrKey, csvName, batch_size=32, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    return data.DataLoader(
        ALI_ImageFolder(root, AttrKey, csvName,
                            transforms.Compose([
                                transforms.Resize((224,224)),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory)

def test_loader(root, AttrKey, csvName, batch_size=1, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    return data.DataLoader(
        ALI_ImageFolder(root, AttrKey, csvName,
                            transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory)

