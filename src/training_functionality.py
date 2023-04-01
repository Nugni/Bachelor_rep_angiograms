import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torchvision.utils

#Function that given relevant parameters perform a training loop of some net
#returns 2 lists of training and val loss for plotting loss as a function
#of training iterations
def trainLoop(net, optimizer, criterion, device, epochs, train_loader, val_loader, print_interv=5):
    train_loss_lst = []
    val_loss_lst = []
    for epoch in range(epochs):
        run_train_loss = 0.0
        run_val_loss = 0.0
        #iterate over training dataset
        for i, data in enumerate(train_loader, 0):
            inputs, labs = data

            #transfer data to device, e.g. GPU
            inputs, labs = inputs.to(device), labs.to(device)

            #maybe
            #inputs = inputs[0]. prolly not. 

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = net(inputs.float()) #should fix double error

            loss = criterion(input=outputs.float(), target=labs.float()) #.float() should fix int error
            loss.backward()
            optimizer.step()

            #print statistics:
            run_train_loss = loss.item()
            if i % print_interv == print_interv-1: #print every 10 minibatches
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, run_train_loss / print_interv))
                train_loss_lst.append(run_train_loss / print_interv)
                running_loss = 0.0

                #compute some val error
                val_input, val_lab = next(iter(val_loader))

                #move to device
                val_input, val_lab = val_input.to(device), val_lab.to(device)

                val_out = net(val_input.float())

                loss_val = criterion(input=val_out.float(), target=val_lab.float())
                run_val_loss = loss_val.item()

                val_loss_lst.append(run_val_loss)
                run_val_loss = 0.0
    print("Finished training! :^)")
    return train_loss_lst, val_loss_lst, net


#function to plot traing loss and possibly val loss after training
def visualizeLoss(model_name, print_interv, train_lst, val_lst=None):
    xTrain = ((np.arange(len(train_lst))) + train_lst[0])*print_interv
    yTrain = train_lst
    plt.plot(xTrain, yTrain, label='Train loss')

    #Add val loss to plot if wished
    if val_lst is not None:
        xVal = ((np.arange(len(val_lst))) + val_lst[0])*print_interv
        yVal = val_lst
        plt.plot(xVal, yVal, label='Val loss')

    #Add text to plot
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.title("Loss of {0} over number of iterations".format(model_name))
    plt.legend() #should plot which graph is which
    plt.show()




# method for testing net
# Check if works with batches
def test_net(net, testLoader, device, illustrate=False):
    testIter = iter(testLoader)
    acum_score = 0
    for i in range(len(testIter)):
        test_input, test_lab_orig = next(testIter)
        #predict
        test_input, test_lab = test_input.to(device), test_lab_orig.to(device)
        #detach from device
        test_out = (net(test_input.float())).cpu().detach() #put output back on cpu
        #formatér
        orig_lab = test_lab_orig.numpy()[0]
        pred_lab = np.array(test_out.numpy()[0] > 0.5).astype(int)
        #compute f1 score
        score = f1_score(orig_lab[0], pred_lab[0], average="micro")
        acum_score += score

        if illustrate:
            print("actual label:")
            plt.imshow(torchvision.utils.make_grid(test_lab_orig).numpy()[0], cmap="gray", vmin=0, vmax=1)
            plt.show()
            print("predicted lab:")
            plt.imshow(torchvision.utils.make_grid(test_out).numpy()[0] > 0.5, cmap="gray", vmin=0, vmax=1)
            plt.show()
        #print f1 score
        print("f1 score: {0:.5f}".format(score))
        print()
    print("Mean f1 score: {0:.5f}".format(acum_score/len(testLoader)))