from src.dataset.dataloader import get_mutated_dataloader
from src.model.Resnet import make_model
from src.model.loss import loss_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy




def get_mean_of_list(L):
    return sum(L) / len(L)

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_training_dataset_mutated = get_mutated_dataloader()

    resnetq=make_model().to(device)
    resnetk = copy.deepcopy(resnetq).to(device)
    optimizer = torch.optim.Adam(resnetq.parameters(),0.0024, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_training_dataset_mutated), eta_min=0,
                                                           last_epoch=-1)
    losses_train = []
    num_epochs = 200
    momentum=0.999
    flag = 0
    K=8192
    queue = None


    if not os.path.exists('results'):
        os.makedirs('results')

    if(os.path.isfile("results/modelq.pth")):
        resnetq.load_state_dict(torch.load("results/modelq.pth"))
        resnetk.load_state_dict(torch.load("results/modelk.pth"))
        optimizer.load_state_dict(torch.load("results/optimizer.pth"))


        temp = np.load("results/lossesfile.npz")
        losses_train = list(temp['arr_0'])
        queue = torch.load("results/queue.pt")


    if queue is None:
        while True:

            with torch.no_grad():
                for (_, sample_batched) in enumerate(dataloader_training_dataset_mutated):

                    xk = sample_batched['image2']
                    xk = xk.to(device)
                    k = resnetk(xk)
                    k = k.detach()

                    k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < K:
                            queue = torch.cat((queue, k), 0)
                        else:
                            flag = 1

                    if flag == 1:
                        break

            if flag == 1:
                break
    resnetq.train()

    for epoch in range(num_epochs):

        print(epoch)

        epoch_losses_train = []

        for (_, sample_batched) in enumerate(dataloader_training_dataset_mutated):

            optimizer.zero_grad()

            xq = sample_batched['image1']
            xk = sample_batched['image2']

            xq = xq.to(device)
            xk = xk.to(device)

            q = resnetq(xq)
            k = resnetk(xk)
            k = k.detach()

            q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
            k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

            loss = loss_function(q, k, queue)

            epoch_losses_train.append(loss.cpu().data.item())

            optimizer.step()
            if epoch >= 10:
                scheduler.step()

            queue = torch.cat((queue, k), 0)

            if queue.shape[0] > K:
                queue = queue[256:,:]

            for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
                θ_k.data.copy_(momentum*θ_k.data + θ_q.data*(1.0 - momentum))

        losses_train.append(get_mean_of_list(epoch_losses_train))

        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train)
        plt.legend(['Training Losses'])
        plt.savefig('losses.png')
        plt.close()

        torch.save(resnetq.state_dict(), 'results/modelq.pth')
        torch.save(resnetk.state_dict(), 'results/modelk.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')
        np.savez("results/lossesfile", np.array(losses_train))
        torch.save(queue, 'results/queue.pt')

if __name__=="__main__":
    train()