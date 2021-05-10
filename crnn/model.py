import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tabulate import tabulate
import time
import datetime
from PIL import Image
import pickle
import ctypes
import sys
import os
import platform
from sklearn.model_selection import KFold

ARCH = "CNN -> RNN"
VER = "2206190500"


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
optimizer2= None

# keep track of initialization state of model
initd = False

# k-fold cross-validation variables
k_num = 10
kfold = None
k_gen = None
kloop = 0

img_width = 32
img_height = 32

input_size = 1024
hidden_dim = 100
external_size = 50 
target_size = 3

# convert data to a normalized torch.FloatTensor
transform = None

# choose the training and test datasets

train_loader = None
test_loader = None
valid_loader = None
classes = []
train_data = None
train_indices = None

# logs epochs' elapsed time and validation loss for every epoch
epochs = []

# keeps track whether epoch stats have been written to file or not
written = False

# path, train and test loaders
p = ""
tr = ""
te = ""

dataset_loaded = False

finalresults = None
traintime = None
params = None

def dataset(path, train, test):
    global p, tr, te, model, batch_size, transform, num_workers, k_num, valid_size, dataset_loaded, kfold, train_data, train_indices
    p = path
    tr = train
    te = test
    if not transform:
        print("Model not initialized, use 'model.init()' to fix this")
        return False
    train_data = datasets.ImageFolder(root=path+train, transform=transform)
    test_data = datasets.ImageFolder(root=path+test, transform=transform)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    train_indices = list(range(num_train))
    np.random.shuffle(train_indices)

    num_test = len(test_data)
    test_indices = list(range(num_test))
    np.random.shuffle(test_indices)
    test_idx = test_indices

    # prepare data loaders (combine dataset and sampler)
    dataset_loaded=True
    krotate()
    test_sampler = SubsetRandomSampler(test_idx)
    global train_loader, valid_loader, test_loader, classes
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                                              num_workers=num_workers)
    classes = train_data.classes
    for i in range(len(classes)):
        classes[i] = classes[i].title()
    global classes_len
    classes_len = len(classes)
    print("Dataset loaded")
    return True


def krotate():
    if not dataset_loaded:
        print("Dataset not loaded, use 'model.dataset()' to fix this")
        return False
    global train_loader, valid_loader, kfold, k_num, train_data, k_gen, train_indices, kloop
    if not kfold:
        kfold = KFold(k_num, True, np.random.randint(low=0, high=100000000))
    if not k_gen or kloop == 10:
        k_gen = kfold.split(train_indices)
        kloop = 0
    data = next(k_gen)
    kloop += 1
    kfold_train = SubsetRandomSampler(data[0])
    kfold_valid = SubsetRandomSampler(data[1])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=kfold_train,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=kfold_valid,
                                               num_workers=num_workers)


class RNN(nn.Module):
    def __init__(self, hidden_dim, input_size, external_size, target_size):
        super(RNN, self).__init__()
        global img_height, batch_size
        self.factor = int(int(int(img_height/2)/2)/2)
        self.factor = self.factor * self.factor * 64
        self.lstm = nn.LSTM(self.factor, hidden_dim)
        self.fc = nn.Linear(hidden_dim, external_size)
        self.output = nn.Linear(external_size, target_size)
        self.dropout = nn.Dropout(0.10)
        self.hidden = hidden_dim

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        y = torch.rand(len(x), self.hidden)
        for i in range(len(x)):
          y[i] = x[i][0]
        x = self.fc(y)
        x = F.softplus(self.output(x), beta=0.5)
        x = self.dropout(x)
        return x


# define the CNN architecture
class Net(nn.Module):
    def __init__(self, output_size, img_size):
        super(Net, self).__init__()
        # date of model creation
        self.date = datetime.datetime.now()
        # keep track of validation loss to prevent loss of training if loaded later on
        self.valid_loss = np.Inf
        # keep track of overall test accuracy
        self.test_accuracy = np.Inf
        self.epoch = 0
        # convolutional layer (sees img_heightximg_widthx3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees img_height/2ximg_width/2x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees img_height/4ximg_width/4x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * img_width/8 * img_width/8  -> 500)
        self.factor = int(int(int(img_size/2)/2)/2)
        self.factor = self.factor * self.factor * 64
        self.fc1 = nn.Linear(self.factor, output_size)
        # dropout layer (p=0.10)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, self.factor)
        # add dropout layer
        x = self.dropout(x)
        return x


# create model variables
model = None
model2 = None

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
    global model, model2, optimizer, optimizer2
    if not model or not model2:
        print("Models not initialized, run 'model.init()' to fix this")
        return False
    cp = path.split('\\')
    cp = cp[len(cp) - 1]
    cp = cp.split('.')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
    model.epoch = checkpoint['epoch']
    model.valid_loss = checkpoint['vloss']
    model.test_accuracy = checkpoint['tloss']
    global loaded, model_name
    loaded = True
    if len(model_name) > 0 and (not model_name == cp[0]):
        print("Overriding set model name; was: %s, " % model_name, end="")
        print("now: %s" % cp[0])
        model_name = cp[0]
    print("Model loaded")


def save():
    global model, model2, optimizer, optimizer2
    torch.save({
        'epoch': model.epoch,
        'model_state_dict': model.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer2_state_dict': optimizer.state_dict(),
        'vloss': model.valid_loss,
        'tloss': model.test_accuracy
    }, get_model_path())


def get_version():
    global VER
    return VER


def version_compare(f_model):
    if not f_model:
        print("No models have been passed as parameter")
        return
    global VER
    VER2 = f_model.get_version()
    if not VER2:
        print("Model passed is either not valid or too old")
        return
    if int(VER) > int(VER2):
        print("This version is newer")
        return True
    elif int(VER) == int(VER2):
        print("These versions are identical")
        return True
    else:
        print("You're using an older version, "+VER2+" is a newer version")
        return False


def write_epoch_stats():
    global epochs, written
    if not epochs:
        print("No recorded stats")
        return
    global model_name
    f = open(model_name+'_epochs_stats', 'w')
    for i in range(len(epochs)):
        f.write('('+str(i)+', '+str(epochs[i][0])+', '+str(epochs[i][1])+', '+str(epochs[i][2])+')\n')
    f.close()
    written = True
    for i in range(len(epochs)):
        if epochs[i][2] == model.valid_loss:
            print("Epoch %d is the one with the lowest validation loss: %f" %(i+1, model.valid_loss))
    return True


def get_stats_dir():
    global written, epochs
    if not epochs:
        print("No recorded stats, train your model first")
        return None
    if not written:
        print("Records have not been written, use 'write_epoch_stats()' to fix this")
        return None
    return get_model_path()+'_epochs_stats'


def set_model_name(name):
    global model_name
    model_name = name
    

def get_model_name():
    global model_name
    return model_name


def get_model_path():
    global model_name
    return os.getcwd() +"\\"+ model_name
    
    
def date_created():
    return model.date.isoformat()


def init():
    global criterion, optimizer, optimizer2, model, model2, train_loader, initd, input_size, hidden_dim, external_size, target_size, transform, img_width
    if not transform:
        transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if not model:
        model = Net(output_size=input_size, img_size=img_width)
    if not model2:
        model2 = RNN(hidden_dim=hidden_dim, input_size=input_size, external_size = external_size, target_size=target_size)
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    if not optimizer2:
        optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0)
    if not train_loader:
        print("Dataset not loaded, use 'model.dataset(path, train_folder, test_folder)' to load a dataset")
    initd = True
    title = "Arch: "+ ARCH + ", Epochs: "+ str(n_epochs) +", Version: "+VER
    os = platform.system() + " " + platform.release()
    pythonver = str(sys.version_info[0])+"."+str(sys.version_info[1])
    welcome = "Running Python "+ pythonver +" on "+ os
    if sys.version_info[0] == 3:
        if platform.system() == "Windows":
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        elif platform.system == "Linux":
            sys.stdout.buffer.write(b'\33]0;'+title.encode()+b'\a')
            sys.stdout.buffer.flush()
    elif sys.version_info[0] == 2:
        if platform.system() == "Windows":
            ctypes.windll.kernel32.SetConsoleTitleA(title)
        elif platform.system() == "Linux":
            sys.stdout.write(b'\33]0;'+title.encode()+b'\a')
            sys.stdout.flush()
    global model_name
    if len(model_name) == 0:
        print("No name has been entered for the new model, generating generic name")
        dtime = datetime.datetime.now()
        model_name = "model_" + VER + "_" + str(dtime.day).zfill(2) + str(dtime.month).zfill(2) + str(dtime.year)[2:] + str(dtime.hour).zfill(2) + str(dtime.minute).zfill(2) + "-" + str(np.random.randint(100, 999))
    print("Creating new model under the name: '%s'" % (model_name))
    print("Model initialized successfully")
    print(welcome)
    
    writeparams()


test_loss = 0.0


def print_network():
    print(model)
    
    
def writeparams():
    global params
    if not params:
        print("Set params first using 'model.setparams(params)'")
        return
    f = open(get_model_name()+"_logs", 'w')
    f.write("params: {\n")
    for i in params:
        f.write("   "+i+": "+str(params[i])+",\n")
    f.write("}\n\n")
    f.close()
    

def logresults():
    global finalresults
    if not finalresults:
        print("Model not tested, run 'model.test()' first then re-run this function")
        return
    f = open(get_model_name()+"_logs", 'a')
    f.write("results: {\n")
    for i in finalresults:
        f.write("   "+i+":"+str(finalresults[i])+",\n")
    f.write("}\n\n")
    f.close()
    print("Results logged in "+get_model_name()+"_logs")


def setparams(crit=None, optimize=None, optimize2=None, num_epochs=None, bat_size=None, val_size=None, trans=None, reload=False, hid_dim=None, in_size=None, ext_size=None, img_size=None):
    print("Setting parameters")
    global transform
    if train_loader and not reload:
        print(" *Dataset already initialized and some parameters will not be in effect until the dataset is reloaded, set 'reload=True' to avoid this", end="")
    if crit:
        global criterion
        criterion = crit
        print(" Criterion set to: ", end="")
        print(criterion)
    if optimize:
        global optimizer
        optimizer = optimize
        print(" Optimizer set to: ", end="")
        print(optimizer)
    if optimize2:
        global optimizer2
        optimizer2 = optimize2
        print(" Optimizer2 set to: ", end="")
        print(optimizer2)
    if num_epochs:
        global n_epochs
        n_epochs = num_epochs
        print(" Number of epochs set to: ", end="")
        print(n_epochs)
    if bat_size:
        global batch_size
        batch_size = bat_size
        print(" Batch size set to: ", end="")
        print(batch_size)
    if val_size:
        global valid_size
        valid_size = val_size
        print(" Validation size set to: ", end="")
        print(valid_size)
    if img_size:
        global img_width, img_height
        img_width = img_size
        img_height = img_size
        transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print(" Image size in transformation set to: ", end="")
        print(transform)
    if hid_dim:
        global hidden_dim
        hidden_dim = hid_dim
        print(" Hidden dimension size set to: ", end="")
        print(hidden_dim)
    if in_size:
        global input_size
        input_size = in_size
        print(" Input size set to: ", end="")
        print(input_size)
    if ext_size:
        global external_size
        external_size = ext_size
        print(" External size set to: ", end="")
        print(external_size)
    if trans and img_size:
        print(" *Parameter 'trans' overriding 'img_size' parameter, set only one of them to fix this")
    if trans:
        transform = trans
        print(" Transform set to: ", end="")
        print(transform)
    if reload:
        global p, tr, te
        dataset(p, tr, te)
        print(" *Dataset reloaded, all parameters in effect")
    global params
    params = {'criterion': crit, 'optimizer': optimize, 'optimizer2': optimize2, 'num_epochs': num_epochs, 'bat_size': bat_size, 'val_size': val_size, 'trans': trans, 'reload': reload, 'hid_dim': hid_dim, 'in_size': in_size, 'ext_size': ext_size, 'img_size': img_size}
    print("Parameter setting complete")


def train():
    global initd, n_epochs, train_loader, valid_loader, criterion, optimizer, optimizer2, model, model2, epochs, written, batch_size, VER
    if not initd:
        print("Model not initialized, run 'model.init()' to fix this")
        return()
    written=False
    epochs = []
    valid_epochs = 0
    since_valid_epoch = 0
    train_start = time.time()
    if not kfold:
        validation = False
        print("Validation disabled")
    else:
        validation = True
    if not loaded:
        print("No models have been loaded, creating a new model")
        save()

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()
        model2.cuda()

    print("Training for %d epochs in batches of %d" %(n_epochs, batch_size))
    avgepoch = 0
    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        print("Epoch #%d" % epoch, end="", flush=True)
        start = time.time()
        model.train()
        model2.train()
 
        for data, target in train_loader:

            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            optimizer2.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            output = model2(output)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            optimizer2.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            model.epoch = epoch

        if validation:
            ######################
            # validate the model #
            ######################
            model.eval()
            model2.eval()
            since_valid_epoch += 1
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                output = model2(output)
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
            epochs.append((epochtime, train_loss, valid_loss))
            avgepoch = ((avgepoch * (epoch-1))+epochtime)/epoch
            estime = avgepoch * (n_epochs - epoch)
            print(': Training Loss: {:.6f} \tValidation Loss: {:.6f} \tElapsed Time: {:d} seconds'.format(train_loss, valid_loss, epochtime))
            krotate()
            # save model if validation loss has decreased
            if valid_loss < model.valid_loss:
                since_valid_epoch = 0
                valid_epochs += 1
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
    global traintime
    traintime = (totalmins, totalsecs)
    print("\nTraining for %d epochs completed in %d minutes and %d seconds\nThe model has improved across %.2f%% of the epochs, last improvement happened %d epochs ago\n" % (n_epochs, totalmins, totalsecs, (valid_epochs/n_epochs)*100, since_valid_epoch))


def test():
    global initd
    if not initd:
        print("Model not initialized, run 'model.init()' to fix this")
        return()
    global test_loss, class_tp, class_total, classes_len, classes, test_loader
    load_model(get_model_path())
    model.eval()
    model2.eval()
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
        output = model2(output)
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
    print('Test Precision (Overall - mAP): %.2f%%' % (100*total_precision))
    print('Test Recall (Overall): %.2f%%' % (100*total_recall))
    print('Test F1 (Overall): %.2f%%\n' % (100*total_f1))
    tables = []
    for i in range(classes_len):
        print('Confusion matrix for %s:' % (classes[i]))
        table = tabulate([["Predicted condition positive", class_tp[i], class_fp[i]], ["Predicted condition negative", class_fn[i], class_tn[i]]], ["Total population", "Condition positive", "Condition negative"], tablefmt="grid")
        print(table, '\n\n')
        tables.append(table)
    global finalresults, traintime
    accuracies, precisions, recalls, f1s = [], [], [], []
    accuracies.append(test_accuracy[0])
    precisions.append(total_precision)
    recalls.append(total_recall)
    f1s.append(total_f1)
    for i in range(len(accuracy)):
        accuracies.append(accuracy[i])
        precisions.append(precision[i])
        recalls.append(recall[i])
        f1s.append(f1[i])
    finalresults = {'accuracy': accuracies, 'precision': precisions, 'recall': recalls, 'F1': f1s, 'train-time': traintime, 'confusion-matrices': tables}
    logresults()
    return test_accuracy[0]
