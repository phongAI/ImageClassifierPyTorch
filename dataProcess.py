import torch
import os
import torchvision.datasets
from torchvision import io,transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import multiprocessing as mp
import time

class Loader():
    @staticmethod
    def load(path,size = (224,224),batch_size = 8,workers=2,train_split=1,shuffle = True):
        mean,std = Loader.calStdMean(path,input_shape=size,batch_size=batch_size)
        if os.path.exists(path):
            #Get transform
            tranform = transforms.Compose([transforms.Resize(size=size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)
                                           ])
            data = torchvision.datasets.ImageFolder(path, transform=tranform)
            class_name = data.classes
            """Split data"""
            if train_split == 1:
                train_data = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=workers,pin_memory=True)
                test_data = None

            elif 0.001<train_split<0.999:
                #Split data
                totalSample = len(data.targets)
                trainSample = int(train_split*totalSample)
                testSample = totalSample - trainSample

                #Load into batches
                train_data,test_data = torch.utils.data.random_split(data,lengths=[trainSample,testSample])
                train_data = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
                test_data = DataLoader(dataset= test_data, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
            else:
                raise Exception("Incorrect split rate")
            return train_data,test_data,class_name
        else:
            raise Exception("Folder is not existed!")

    @staticmethod
    def getOptimalNumWorkers(path=None,size = (224,224),batch_size = 8,epochs=3):
        if path is not None:
            for i in range(2,mp.cpu_count()+1,2):
                data = Loader.load(path, size=size, batch_size=batch_size, workers=i)
                startTime = time.time()
                for epoch in range(epochs):
                    for image in data:
                        pass
                duration = time.time() - startTime
                print(f"Time for loader is {round(duration,2)}s. Batch_size: {batch_size}. Workers: {i}")
        else:
            raise Exception("Folder is not existed!")



    @staticmethod
    def calStdMean(path,input_shape,batch_size):
        if os.path.exists(path):
            """Load data"""
            dataset = ImageFolder(root=path, transform=transforms.Compose([transforms.Resize(size=input_shape),
                                                                           transforms.ToTensor()
                                                                           ]))
            dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            channels_sum, channel_squared_sum, num_batches = 0, 0, 0

            """Calculate mean and standard value"""
            for data, _ in dataLoader:
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
                num_batches += 1
            mean = channels_sum / num_batches
            std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5

            """Print it out"""
            print(f"Mean value is : {mean.numpy()}")
            print(f"Std value is : {std.numpy()}")
            return mean, std
        else:
            raise Exception("Folder is valid!")



