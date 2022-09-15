import torch
import random
import torchvision
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import random_split as random_split


def data_reorgnize(task_data_loader, total_task_batches=10):
    """
    split the task batches into a list of tasks, each task contains train data and test data
    """
    tasks_list_train = []
    tasks_list_test = []
    # load the tasks
    task_batches = task_data_loader.get_test_batches(total_batches=total_task_batches, augment_images=False)
    for _, data_batch in enumerate(task_batches):
        x_support_set, x_target_set, y_support_set, y_target_set, _ = data_batch
        x_support_set = torch.Tensor(x_support_set).float()
        x_target_set = torch.Tensor(x_target_set).float()
        y_support_set = torch.Tensor(y_support_set).long()
        y_target_set = torch.Tensor(y_target_set).long()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            n, s, c, h, w = x_target_set_task.shape  # n:class_num, s:sample_num_per_class
            print('{} class, {} samples, {} channels, image high {}, width {}'.format(n, s, c, h, w))
            x_support_set_task = x_support_set_task.view(-1,  c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            train_data = (x_support_set_task, y_support_set_task)
            test_data = (x_target_set_task, y_target_set_task)
            tasks_list_train.append(train_data)
            tasks_list_test.append(test_data)
    #print(x_support_set_task[0])

    return tasks_list_train, tasks_list_test


def data_load_1():
    # Data
    print('==>  data loading..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_k_fold_data(k, i, X, y):
    """
        return ith fold train and val data (for k-folds cross-validation)
    """

    index = [i for i in range(len(X))]  # shuffle the data
    random.shuffle(index)
    X, y = X[index], y[index]

    assert k > 1
    # print('the size of training data before train-val split is: ', X.shape[0])
    fold_size = X.shape[0] // k  # data num / fold num

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)
        X_part, y_part = X[idx, :], y[idx]    # idx is the index of validation dataset
        if j == i:  # the ith validation data
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid


class MyDataset(data.Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


class DatasetPerTask(data.Dataset):
    """
    build dataset
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):  #
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def dataset_builder(data, split_val=False, i=1, k=5):
    if split_val:
        x_train, y_train, x_valid, y_valid = get_k_fold_data(k, i, data[0], data[1])
        val_dataset = DatasetPerTask(x_valid, y_valid)
    else:
        x_train, y_train = data[0], data[1]
        val_dataset = None
    train_dataset = DatasetPerTask(x_train, y_train)

    return train_dataset, val_dataset


def data_loader(data, batch_size, shuffle, split_val=False, i=1, k=5):
    """
    build dataloader for training
    param: data: the input, can be a 'dataset' type or list of tensor
    param: batch_size: the batch size of the output dataloader
    param: shuffle: a flag indicating shuffle or not
    param: split_val: a flag indicating whether to split a part of the data as validating data
    param: i, k: related to the cross validation procedure, refer to the doc of func 'get_k_fold_data'
    """
    #if not 'Dataset' in str(type(data)):
        #train_dataset, val_dataset = dataset_builder(data, split_val=split_val, i=i, k=k)

    if split_val == True:
        data_length = len(data)
        train_data_length = int(data_length * 0.95)
        train_dataset, val_dataset = random_split(data, [train_data_length, data_length - train_data_length])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_dataset = data
        val_loader = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader
