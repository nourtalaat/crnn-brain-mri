import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tabulate import tabulate
import time
import datetime
from PIL import Image
import pickle

# default parameters

train_on_gpu = torch.cuda.is_available()

# number of subprocesses to use for data loading

num_workers = 0

# how many samples per batch to load

batch_size = 1

# model name
model_name = ""

# number of epochs to train the model

n_epochs = 100

criterion = None

optimizer = None

# keep track of initialization state of model
initd = False

# percentage of training set to use as validation

valid_size = 0.2

img_width = 32
img_height = 32

# convert data to a normalized torch.FloatTensor

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets

train_loader = None
test_loader = None
valid_loader = None
classes = []

p = ""
tr = ""
te = ""


def dataset(path, train, test):
    global p, tr, te, model
    if not model:
        model = Net()
    p = path
    tr = train
    te = test
    train_data = datasets.ImageFolder(root=path+train, transform=transform)
    test_data = datasets.ImageFolder(root=path+test, transform=transform)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    train_indices = list(range(num_train))
    np.random.shuffle(train_indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = train_indices[split:], train_indices[:split]

    num_test = len(test_data)
    test_indices = list(range(num_test))
    np.random.shuffle(test_indices)
    test_idx = test_indices

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # prepare data loaders (combine dataset and sampler)
    global train_loader, valid_loader, test_loader, classes
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                                              num_workers=num_workers)
    classes = train_data.classes
    for i in range(len(classes)):
        classes[i] = classes[i].title()
    global classes_len
    classes_len = len(classes)


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # date of model creation
        self.date = datetime.datetime.now()
        # keep track of validation loss to prevent loss of training if loaded later on
        self.valid_loss = np.Inf
        # keep track of overall test accuracy
        self.test_accuracy = 0
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 4)
        # dropout layer (p=0.10)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        return x


# instantiate a CNN
model = None

loaded = False


def load_meta():
    global model_name, model
    f = open(model_name+'.pckl', 'rb')
    meta = pickle.load(f)
    f.close()
    model.valid_loss = meta[0]
    if meta[1]:
        model.test_accuracy = meta[1]


def load_model(path):
    global model
    if not model:
        model = Net()
    model.load_state_dict(torch.load(path))
    global loaded, model_name
    loaded = True
    cp = path.split('\\')
    cp = cp[len(cp) - 1]
    cp = cp.split('.')
    if len(model_name) > 0:
        print("Overriding set model name; was: %s, " % model_name, end="")
    model_name = cp[0]
    if len(model_name) > 0:
        print("now: %s" % model_name)
    print("Model loaded: ")
    load_meta()
    print(" Recorded overall test accuracy: %.2f%%" % model.test_accuracy)
    return True


def set_model_name(name):
    global model_name
    model_name = name


def date_created():
    return model.date.isoformat()


def init():
    global criterion, optimizer, model, train_loader, initd
    if not model:
        model = Net()
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    if not train_loader:
        print("Dataset not loaded, use 'model.dataset(path, train_folder, test_folder)' to load a dataset")
    initd = True
    print("Model initialized successfully")


test_loss = 0.0


def print_network():
    print(model)


def setparams(crit=None, optimize=None, num_epochs=None, bat_size=None, val_size=None, trans=None, reload=False):
    if train_loader and not reload:
        print("Dataset already initialized and some parameters will not be in effect until the dataset is reloaded, set 'reload=True' to avoid this")
    if crit:
        global criterion
        criterion = crit
    if optimize:
        global optimizer
        optimizer = optimize
    if num_epochs:
        global n_epochs
        n_epochs = num_epochs
    if bat_size:
        global batch_size
        batch_size = bat_size
    if val_size:
        global valid_size
        valid_size = val_size
    if trans:
        global transform
        transform = trans
    if reload:
        global p, tr, te
        dataset(p, tr, te)
        print("Dataset reloaded, all parameters in effect")


def save():
    global model
    torch.save(model.state_dict(), model_name + '.pt')
    f = open(model_name + '.pckl', 'wb')
    metad = [model.valid_loss, model.test_accuracy]
    pickle.dump(metad, f)
    f.close()
    model.load_state_dict(torch.load(model_name + '.pt'))
    load_meta()


def train():
    global initd
    if not initd:
        print("Model not initialized, run 'model.init()' to fix this")
        return()
    train_start = time.time()
    if valid_size == 0:
        validation = False
        print("Validation disabled")
    else:
        validation = True
    if not loaded:
        print("No models have been loaded, creating a new model")
        global model_name
        if len(model_name) == 0:
            print("No name has been entered for the new model, generating generic name")
            date = model.date.isoformat()
            model_name = "model_" + date[2:10] + "-" + str(np.random.randint(500, 999))
        print("Creating new model under the name: '%s'" % (model_name+'.pt'))
        save()

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    print("Training started")
    avgepoch = 0
    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        print("Epoch #%d" % epoch, end="")
        start = time.time()
        model.train()

        for data, target in train_loader:

            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)

        if validation:
            ######################
            # validate the model #
            ######################
            model.eval()
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)

            # calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)

            # print training/validation statistics
            end = time.time()
            epochtime = int(end-start)
            avgepoch = ((avgepoch * (epoch-1))+epochtime)/epoch
            estime = avgepoch * (n_epochs - epoch)
            print(': Training Loss: {:.6f} \tValidation Loss: {:.6f} \tElapsed Time: {:d} seconds'.format(train_loss, valid_loss, epochtime))
            # save model if validation loss has decreased
            if valid_loss <= model.valid_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(model.valid_loss, valid_loss), end="")
                model.valid_loss = valid_loss
                save()
                print(" Model saved.")
            if estime > 60:
                estmins = int(estime/60)
                estsecs = int(((estime/60)-estmins)*60)
                print("Estimated time to complete training: %d minutes and %d seconds" % (estmins, estsecs))
            elif estime < 60 and int(estime) > 0:
                print("Estimated time to complete training: %d seconds" % (int(estime)))
        else:
            end = time.time()
            epochtime = int(end-start)
            avgepoch = ((avgepoch * (epoch-1))+epochtime)/epoch
            estime = avgepoch * (n_epochs - epoch)
            print(': Training Loss: {:.6f} \tElapsed Time: {:d} seconds'.format(train_loss, int(end-start)), end="")
            save()
            print(" Model saved.")
            if estime > 60:
                estmins = int(estime/60)
                estsecs = int(((estime/60)-estmins)*60)
                print("Estimated time to complete training: %d minutes and %d seconds" % (estmins, estsecs))
            else:
                print("Estimated time to complete training: %d seconds" % (int(estime)))
    global test_loss, class_tp, class_total
    test_loss = 0.0
    class_tp = list(0. for i in range(classes_len))
    class_total = list(0. for i in range(classes_len))
    train_end = time.time()
    totalmins = int((train_end-train_start)/60)
    totalsecs = int((train_end-train_start) - (totalmins*60))
    print("\nTraining completed in %d minutes and %d seconds\n" % (totalmins, totalsecs))


def test():
    global initd
    if not initd:
        print("Model not initialized, run 'model.init()' to fix this")
        return()
    global test_loss, class_tp, class_total, classes_len, classes
    model.eval()
    # iterate over test data
    class_fp = list(0. for i in range(classes_len))
    class_fn = list(0. for i in range(classes_len))
    class_tn = list(0. for i in range(classes_len))
    print("Testing started\n")
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        # print(pred, target.data.view_as(pred))  # pred is a list of batch_size length, same goes for target, if pred[i] == target[i] then prediction is correct
        if not len(correct_tensor) == 1:
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        else:
            correct = correct_tensor
        # calculate test accuracy for each object class
        for i in range(len(target.data)):
            if not pred[i] == target.data.view_as(pred)[i]:
                class_fp[pred[i]] += 1
                class_fn[target.data.view_as(pred)[i]] += 1
            label = target.data[i]
            class_tp[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    # print('Test Loss: {:.6f}\n'.format(test_loss))
    accuracy = list(0. for i in range(classes_len))
    precision = list(0. for i in range(classes_len))
    recall = list(0. for i in range(classes_len))
    f1 = list(0. for i in range(classes_len))
    for i in range(len(classes)):
        print('Results for \"%s\":' % (classes[i]))
        if class_total[i] > 0:
            accuracy[i] = 100 * class_tp[i] / class_total[i]
            class_tn[i] = np.sum(class_total) - (class_tp[i]+class_fn[i]+class_fp[i])
            print(' Accuracy: %2d%% (%2d/%2d)' % (accuracy[i], np.sum(class_tp[i]), np.sum(class_total[i])))
            if not accuracy[i] == 0:
                precision[i] = (class_tp[i]/(class_tp[i]+class_fp[i]))
                recall[i] = (class_tp[i]/(class_tp[i]+class_fn[i]))
                f1[i] = 2 * ((precision[i]*recall[i])/(precision[i]+recall[i]))
                print(' Precision: %.2f%%' % (100*precision[i]))
                print(' Recall: %.2f%%' % (100*recall[i]))
                print(' F1: %.2f%%\n' % (100*f1[i]))
            else:
                print(' Precision: 0%%')
                print(' Recall: 0%%')
                print(' F1: 0%%\n')
        else:
            print(' Accuracy: N/A (no training examples)')
            print(' Precision: N/A (no training examples)')
            print(' Recall: N/A (no training examples)')
            print(' F1: N/A (no training examples)\n')

    test_accuracy = 100. * np.sum(class_tp) / np.sum(class_total), np.sum(class_tp), np.sum(class_total)
    model.test_accuracy = test_accuracy[0]
    save()
    print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % test_accuracy)

    # precision = class_tp/(class_tp+class_fp)
    # recall = class_tp/(class_tp+class_fn)
    # f1 = 2 * ((precision*recall)/(precision+recall))
    actual_classes_len = 0
    for i in range(classes_len):
        if not class_total[i] == 0:
            actual_classes_len += 1
    total_precision = sum(precision)/actual_classes_len
    total_recall = sum(recall)/actual_classes_len
    total_f1 = sum(f1)/actual_classes_len
    print('Test Precision (Overall): %.2f%%' % (100*total_precision))
    print('Test Recall of (Overall): %.2f%%' % (100*total_recall))
    print('Test F1 of (Overall): %.2f%%\n' % (100*total_f1))

    for i in range(classes_len):
        print('Confusion matrix for %s:' % (classes[i]))
        table = tabulate([["Predicted condition positive", class_tp[i], class_fp[i]], ["Predicted condition negative", class_fn[i], class_tn[i]]], ["Total population", "Condition positive", "Condition negative"], tablefmt="grid")
        print(table, '\n\n')

