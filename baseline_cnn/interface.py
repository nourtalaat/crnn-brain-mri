import model

path = input("Dataset path: ")
path = path+"\\"
train = input("Train folder name: ")
test = input("Test folder name: ")

model.setparams(num_epochs=2)
model.dataset(path, train, test)
model.load_model('model_save_test.pt')
model.init()
model.train()
model.test()
