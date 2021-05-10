import model
import plotter



"""
path = input("Dataset path: ")
path = path+"\\"
train = input("Train folder name: ")
test = input("Test folder name: ")
"""

path = "E:\\AAST Portfolio\\Semester 8\\Project 2\\1512427\\preprocessing\\processed\\complete\\separated\\upscaling\\"
train = "train_upscaled"
test = "test"

model.setparams(num_epochs=200, bat_size=1)
model.init()
model.dataset(path, train, test)
model.train()
model.test()

model.write_epoch_stats()
plotter.plot(model.get_stats_dir())
plotter.save(model.get_model_path())

print("End of run for version: "+model.get_version())
input("Press enter to exit..")
