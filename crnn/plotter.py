import matplotlib.pyplot as plt


fig = None

# reads file and creates plots
def plot(path):
    if not path:
        print("No valid path has been given")
        return
    global fig
    stats = []
    with open(path) as f:
        for line in f:
            line= line.rstrip('\n')
            line = line[1:len(line)-1]
            line = line.split(',')
            line = (int(line[0]), float(line[1]), float(line[2]), float(line[3]))
            stats.append(line)
    times = []
    tloss = []
    vloss = []
    epochs = []
    for i in stats:
        epochs.append(i[0])
        times.append(i[1])
        tloss.append(i[2])
        vloss.append(i[3])
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 1])
    ax.plot(epochs, vloss, 'b', label="Validation Loss", color="green")
    ax.plot(epochs, tloss, 'b', label="Training Loss", color="blue")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Across Epochs')
    plt.legend()

# saves plots as PNG files with given name
def save(path):
    if not path:
        print("No valid path has been given")
        return
    global fig
    if not fig:
        print("No stats loaded, use 'plot(path)' to fix this")
        return False
    fig.savefig(path+'-loss.png')
    return True