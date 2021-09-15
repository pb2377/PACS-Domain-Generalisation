import os.path as osp

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

ROOTS = {'pacs': "../Datasets/PACS/Raw-images/kfold",
         'domainnet': "../Datasets/DomainNet"}


class BaseDataset(data.Dataset):
    def __init__(self, root, list_ids, labels, transform):
        super(BaseDataset, self).__init__()
        self.data_path = root
        self.list_ids = list_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        imgpath = osp.join(self.data_path, self.list_ids[index])
        img = Image.open(imgpath).convert('RGB')
        lbl = int(self.labels[index])
        return imgpath, self.transform(img), lbl


def get_dataloaders(args):
    train_transform, test_transform = get_transforms(args)

    root = ROOTS['domainnet' if args.domainnet else 'pacs']

    # train datasets
    train_names, train_labels, val_names, val_labels = [], [], [], []
    for dname in args.sources:
        train_names, train_labels = dataset_info(args, dname, 'train', path_list=train_names, label_list=train_labels)
        val_names, val_labels = dataset_info(args, dname, 'crossval', path_list=val_names, label_list=val_labels)

    train_dataset = BaseDataset(root, train_names, train_labels, transform=train_transform)
    val_dataset = BaseDataset(root, val_names, val_labels, transform=test_transform)

    # test dataset
    test_names, test_labels = dataset_info(args, args.target, 'test', path_list=[], label_list=[])
    test_dataset = BaseDataset(root, test_names, test_labels, transform=test_transform)

    # Domain adaptation data
    if args.adapt:
        raise NotImplementedError

    # Concat datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)

    test_loaders = {
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True, drop_last=False),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    }

    return train_loader, test_loaders


def get_transforms(args):
    # Train Transform
    p_grey = 0.1
    train_tr = [
        transforms.RandomResizedCrop((IMAGE_SIZE), (0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter,
                               saturation=args.jitter, hue=min(0.5, args.jitter)),
        transforms.RandomGrayscale(p=p_grey),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    train_tr = transforms.Compose(train_tr)

    # Test Transform
    test_tr = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_tr, test_tr


def dataset_info(args, dname, dsplit, path_list, label_list):
    if args.domainnet:
        # domainnet
        txt_path = osp.join(osp.dirname(__file__), 'data', 'domainnet', '%s_%s.txt' % (dname, dsplit))
    else:
        # pacs
        txt_path = osp.join(osp.dirname(__file__), 'data', 'pacs', '%s_%s_kfold.txt' % (dname, dsplit))
    assert osp.exists(txt_path)

    df = pd.read_csv(txt_path, delimiter=' ', header=None, names=['image_path', 'idx_label'])
    # convert labels to ints
    df['idx_label'] = df['idx_label'].astype(int)
    df['idx_label'] -= min(df['idx_label'])

    path_list.extend(df['image_path'].astype(str).tolist())
    label_list.extend(df['idx_label'].tolist())
    return path_list, label_list
