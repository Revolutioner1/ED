import torch
import torchvision
from torchvision import transforms
from torch.utils.data import  DataLoader


def get_cifar100_dataloaders(id,num_class , batch_size,num_workers):
    root = './data'
    train_cifar100, val_cifar100 = instance_cifar100(root)
    if id == 0 :
        indices = torch.where(torch.tensor(train_cifar100.targets) < num_class[id])[0]
        train_dataset = torch.utils.data.Subset(train_cifar100, indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                                             ,num_workers=num_workers)
        indices = torch.where(torch.tensor(val_cifar100.targets) < num_class[id])[0]
        val_dataset = torch.utils.data.Subset(val_cifar100, indices)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True
                                                   , num_workers=num_workers)

    else:
        indices = torch.where((torch.tensor(train_cifar100.targets.targets) > num_class[id]) &
                              (torch.tensor(second_train_dataset_CIF100.targets) < num_class[id+1]))[0]
        train_dataset = torch.utils.data.Subset(train_cifar100, indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                                   , num_workers=num_workers)

        indices = torch.where((torch.tensor(train_cifar100.targets.targets) > num_class[id]) &
                              (torch.tensor(second_train_dataset_CIF100.targets) < num_class[id+1]))[0]
        val_dataset = torch.utils.data.Subset(val_cifar100, indices)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True
                                                 , num_workers=num_workers)

    return train_loader,val_loader
def instance_cifar100(root):
    train_dataset = torchvision.datasets.CIFAR100(
        root = root,
        train = True,
        transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
        download=True
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root = root,
        train = True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
        download=True
    )
    return train_dataset,val_dataset

first_train_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'./data',
    train=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    download=False
)

first_test_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'./data',
    train=False,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
    download=False
)
train_loader = torch.utils.data.DataLoader(first_train_dataset_CIF100, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(first_test_dataset_CIF100, batch_size=4, shuffle=True)

indices = torch.where(torch.tensor(first_train_dataset_CIF100.targets) <= 32)[0]
first_train_dataset_new = torch.utils.data.Subset(first_train_dataset_CIF100, indices)
first_train_loader_new = torch.utils.data.DataLoader(first_train_dataset_new, batch_size=4, shuffle=True)

indices = torch.where(torch.tensor(first_test_dataset_CIF100.targets) <= (32))[0]
first_test_dataset_new = torch.utils.data.Subset(first_test_dataset_CIF100, indices)
first_test_loader_new = torch.utils.data.DataLoader(first_test_dataset_new, batch_size=4, shuffle=True)


#  second teacher

second_train_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'./data',
    train=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    download=False
)

second_test_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'./data',
    train=False,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
    download=False
)

indices = torch.where((torch.tensor(second_train_dataset_CIF100.targets) > 32) & (torch.tensor(second_train_dataset_CIF100.targets) <= 65))[0]
second_train_dataset_CIF100 = torch.utils.data.Subset(second_train_dataset_CIF100, indices)
second_train_loader_new = torch.utils.data.DataLoader(second_train_dataset_CIF100, batch_size=4, shuffle=True)

indices = torch.where((torch.tensor(second_test_dataset_CIF100.targets) > 32) & (torch.tensor(second_test_dataset_CIF100.targets) <= 65))[0]
second_test_dataset_CIF100 = torch.utils.data.Subset(second_test_dataset_CIF100, indices)
second_test_loader_new = torch.utils.data.DataLoader(second_test_dataset_CIF100, batch_size=4, shuffle=True)


#  third teacher

third_train_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'C:\Users\ydmyANDxy\PycharmProjects\202407\datasets\dataset_CIFAR100',
    train=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    download=False
)

third_test_dataset_CIF100 = torchvision.datasets.CIFAR100(
    root=r'C:\Users\ydmyANDxy\PycharmProjects\202407\datasets\dataset_CIFAR100',
    train=False,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
    download=False
)

indices = torch.where((torch.tensor(third_train_dataset_CIF100.targets) > 65) & (torch.tensor(third_train_dataset_CIF100.targets) <= 99))[0]
third_train_dataset_CIF100 = torch.utils.data.Subset(third_train_dataset_CIF100, indices)
third_train_loader_new = torch.utils.data.DataLoader(third_train_dataset_CIF100, batch_size=4, shuffle=True)

indices = torch.where((torch.tensor(third_test_dataset_CIF100.targets) > 65) & (torch.tensor(third_test_dataset_CIF100.targets) <= 99))[0]
third_test_dataset_CIF100 = torch.utils.data.Subset(third_test_dataset_CIF100, indices)
third_test_loader_new = torch.utils.data.DataLoader(third_test_dataset_CIF100, batch_size=4, shuffle=True)