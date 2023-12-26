
import numpy as np
import torch
import torch.nn as nn


# a function that trains the model
def trainModel(
    ANNmodel, train_loader, test_loader, lossfun=nn.BCEWithLogitsLoss, optimizer=torch.optim.SGD, learningRate=0.01, numepochs=1000
):
    '''
    ANNmodel: model to train
    train_loader: dataloader for training data
    test_loader: dataloader for test data
    lossfun: loss function (default: nn.BCEWithLogitsLoss)
    optimizer: optimizer (default: torch.optim.SGD)
    learningRate: learning rate (default: 0.01)
    numepochs: number of epochs (default: 1000)
    
    returns: trainAcc, testAcc, losses
    trainAcc: list of training accuracies (1 per epoch)
    testAcc: list of test accuracies (1 per epoch)
    losses: list of losses (1 per epoch)
    
    '''
    # loss function and optimizer
    lossfun = lossfun()
    optimizer = optimizer(ANNmodel.parameters(), lr=learningRate)

    # initialize losses
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []

    # loop over epochs
    for epochi in range(numepochs):

        # loop over training data batches
        batchAcc = []
        batchLoss = []
        for X, y in train_loader:
            # forward pass and loss
            yHat = ANNmodel(X)
            loss = lossfun(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute training accuracy for this batch
            batchAcc.append(100 * torch.mean(((yHat > 0) == y).float()).item())
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        X, y = next(iter(test_loader))  # extract X,y from test dataloader

        yHat = ANNmodel(X)
            
        testAcc.append(100 * torch.mean(((yHat > 0) == y).float()).item())

    # function output
    return trainAcc, testAcc, losses
