

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stock_dataset_dwt import LoadData
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np


from gat import GATNet

import warnings
warnings.filterwarnings('ignore')




def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_data = LoadData(data_path=["./stock_db4.npy", "./stock_characteristic.npy"], num_nodes=29, divide_days=[3352,250],
                         time_interval=1, history_length=7,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)  # num_workers是加载数据（batch）的线程数目




    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



    training_loss = []

    my_net = GATNet(in_c=7 , hid_c=32, out_c=1, n_heads=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)


    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters(),lr=1e-4)


    Epoch = 2000

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        count = 0
        start_time = time.time()

        for data in train_loader:
            my_net.zero_grad()
            count +=1
            predict_value = my_net(data, device).to(torch.device("cpu"))

            loss= criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()


            loss.backward()

            optimizer.step()
        training_loss.append(epoch_loss)
        end_time = time.time()
        if (epoch+1)%50==0:
            torch.save(my_net.state_dict(),'./net_params_'+str(epoch)+'.pth')


        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss,
                                                                          (end_time - start_time) / 60))




if __name__ == '__main__':
    torch.set_num_threads(5)
    main()
