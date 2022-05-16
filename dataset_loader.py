import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def dataloader(args,download=False):
    train_resolution = 299 if args.arch == "inceptionv3" else 224
    test_resolution = (342, 299) if args.arch == 'inceptionv3' else (256, 224)
    if args.ds == 'imagenet':
        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(train_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(test_resolution[0]),
            transforms.CenterCrop(test_resolution[1]),
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = args.data
        data_root = os.path.join(dir_path, 'data', str(args.ds))

        train_dataset = datasets.CIFAR10(data_root, download=download, train=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(data_root, download=download, train=False, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    dataset_length = int(len(train_dataset) * args.data_percentage)
    if args.data_percentage == 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                                 [dataset_length,
                                                                  len(train_dataset) - dataset_length])
        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # evaluate on validation set
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader,train_sampler,dataset_length
