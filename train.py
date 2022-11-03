from dataProcess import Loader
from modelStructure import Resnet
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
folder = "../../Dataset/PetImages"
input_shape = (224,224)
batch_size = 128
workers = 4
classes = 2
epochs = 25
lr = 2e-4
weight_decay = 3e-4

def main():
    torch.cuda.empty_cache()
    """Load data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, test_data,class_name = Loader.load(path=folder,size=input_shape,batch_size=batch_size,workers=workers,train_split=0.8)
    resnet = Resnet(classes=len(class_name)).to(device)
    resnet.train_and_eval(train_data=train_data,test_data=test_data,epochs=epochs,learning_rate=lr,weight_decay=weight_decay)
    #resnet.loadCheckpoint(path="checkpoints/145407/Checkpoint_best_epoch3.pth")
    # plt.imshow(img_show)
    # plt.show()
    # val_loss,val_acc = resnet.evaluate(train_data)
    # print(val_loss)
    # print(val_acc)

if __name__ == '__main__':
    main()

