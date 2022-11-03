import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision.models import ResNet34_Weights
from tqdm import tqdm
checkPointFolder = 'checkpoints'

class Resnet(nn.Module):
    def __init__(self,classes = 2):
        super(Resnet, self).__init__()
        self.classes = classes
        self.model = self.__createfeatureExtractor()
        self.model.fc = self.__createFullConnected()
        """Initialize weight"""
        self.weight_initialize()

        """Select device"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        """Default Optimizer"""
        # optimizer
        #self.optimizer = optim.SGD(params=self.model.parameters(), lr=1e-2,momentum=0.9)

        """Create checkpoint folder"""
        os.makedirs(checkPointFolder,exist_ok=True)

        """Loss"""
        self.criterion = nn.CrossEntropyLoss()

    def weight_initialize(self):
        for layer in self.model.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data)

    def __createfeatureExtractor(self):
        model = models.resnet34(weights = ResNet34_Weights.DEFAULT)
        for layer in model.parameters():
            layer.requires_grad = False
        return model

    def __createFullConnected(self):
        sequence = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128,out_features=self.classes)
        )
        return sequence

    def forward(self,x):
        x = self.model(x)
        return x

    def saveCheckPoint(self,path,loss,epoch,monitor = "val_loss"):
        dest_path = os.path.join(path,f"Checkpoint_best_epoch{epoch}.pth")
        name = ""
        if len(loss) == 0:
            raise Exception("Not exist loss")
        elif len(loss) == 1:
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            # }, dest_path)
            torch.save(self.model, dest_path)
            print(f"Save best Checkpoint: {os.path.basename(dest_path)}")
            name = dest_path
        else:
            if min(loss) == loss[-1]:
                # torch.save({
                #     'model_state_dict': self.model.state_dict(),
                #     'optimizer_state_dict': self.optimizer.state_dict(),
                # }, dest_path)
                # print(f"Save best Checkpoint:{dest_path}")
                torch.save(self.model, dest_path)
                print(f"Save best Checkpoint: {os.path.basename(dest_path)}")
                name = dest_path
        return name
        # name = os.path.basename(path)
        # print(f"\nSave check point: {name}!")
        #torch.save(self.model, path)

    def loadCheckpoint(self,path):
        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.load_state_dict(checkpoint['optimizer_state_dict'])
        # loss = checkpoint['loss']
        self.model = torch.load(path)

    def train_and_eval(self,train_data=None,test_data=None,learning_rate = 1e-3,epochs= 20,weight_decay = 3e-4):
        #Time
        time = datetime.datetime.now().strftime('%H%M%S')
        checkPointPath = os.path.join(checkPointFolder,time)
        os.makedirs(checkPointPath,exist_ok=True)
        scaler = torch.cuda.amp.GradScaler()
        # optimizer
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=0.9,nesterov=True)
        #self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #Learning rate reducer
        lr_reducer = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=0.5,patience=2)
        #lmbda = lambda epochs: 0.95 ** epochs
        #lr_reducer = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,lr_lambda=lmbda)


        if train_data is not None:
            train_loss_record = []
            val_loss_record = []
            list_Checkpoint = []
            self.model.train()
            total_sample = len(train_data.sampler)
            batch = len(train_data)
            print(f"Data for train: {total_sample}")
            for epoch in range(epochs):
                loops = enumerate(tqdm(train_data,total=batch,desc=f"Epoch [{epoch+1}/{epochs}",unit="batch"))
                current_loss = 0
                for i,(image,label) in loops:
                    self.optimizer.zero_grad()
                    with torch.amp.autocast(device_type=self.device,enabled=False):
                        image = image.to(self.device)
                        label = label.to(self.device)
                        output = self.model(image)
                        loss = self.criterion(output,label)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    current_loss += loss.item()*image.size(0)
                train_loss_record.append(current_loss/total_sample)

                #Calculate loss and evaluate data
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                train_loss = round(sum(train_loss_record)/len(train_loss_record),4)

                if test_data is not None:
                    # Valuation
                    val_loss, val_acc = self.evaluate(test_data)
                    val_loss_record.append(val_loss)
                    lr_reducer.step(val_loss)
                    print(f"\nTrain loss: {train_loss}. Val loss: {val_loss}. Val accuracy: {str(val_acc)}%. Current lr: {current_lr}.")
                else:
                    print(f"\nTrain loss: {train_loss}")
                """lr_reducer.step(val_loss)"""

                #Save check point
                name_checkpoint = self.saveCheckPoint(checkPointPath,val_loss_record,epoch+1)
                #Remove unecessary best
                if len(name_checkpoint)>0:
                    list_Checkpoint.append(name_checkpoint)
                    if len(list_Checkpoint)>1:
                        oldPath = list_Checkpoint[-2]
                        os.remove(oldPath)

    def predict(self,image):
        with torch.no_grad():
            image = image.to(self.device)
            self.model.eval()
            prediction = self.model(image)
            prob = torch.softmax(prediction,dim=1)
            index = torch.argmax(prob)
        return prob,index

    def evaluate(self,dataset):
        if dataset is not None:
            val_loss = 0
            total_correct = 0
            total_image = len(dataset.sampler)
            with torch.no_grad():
                if self.model is not None:
                    self.model.eval()
                    for (image,label) in dataset:
                        output = self.model(image.to(self.device))
                        label = label.to(self.device)
                        predictions = torch.argmax(output,1)
                        loss = self.criterion(output,label)
                        val_loss += loss.item()*image.size(0)
                        total_correct += (predictions == label).sum()
                    val_loss = round(val_loss/total_image,4)
                    probs = 100 * (total_correct / total_image)
                    val_accu = round(float(probs.cpu()),1)
            return val_loss,val_accu



