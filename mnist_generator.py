import numpy as np
import os as os
import backprop as bp
import tensorflow as tf
import matplotlib.pyplot as plt

def OneHotEncoder(data):
    onehot = np.zeros((data.size, 10))
    for i in range(0, data.size):
        onehot[i, int(data[i])] = 1
    return onehot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

train = np.genfromtxt("mnist_train.csv", delimiter = ",")
test = np.genfromtxt("mnist_test.csv", delimiter = ",")
print("Loading finished")
train_label = train[:, 0]
train_label = OneHotEncoder(train_label)

test_label = test[:, 0]
test_label = OneHotEncoder(test_label)
print('converted labels')
hidden_layers = [1024, 256]
model = bp.backprop(train[:, 1:]/255, train_label, 2, hidden_layers, max_iter = 1000, eta = 0.002, f_type = "C", b_type = "R", b = 100, disp_step = 10)
model.fit()
input_num = 0
while(input_num != 'E' or input_num != 'e'):
    input_num = str(input("Enter a number, press E to exit: "))
    print("Input was : " + str(input_num))
    size_ = len(input_num)
    input_num = int(input_num)
    input_list = []
    for i in range(0, size_):
        input_list.append(input_num % 10)
        input_num = input_num - (input_num % 10)
        input_num = input_num / 10
    input_list = np.asarray(input_list)
    input_list = np.flip(input_list)
    input_ = np.zeros((input_list.size, 10))
    for i in range(0, input_list.size):
        input_[i, input_list[i].astype(int)] = 1
    image = model.predict(input_, direction = "B")
    disp = np.zeros((28, size_*28))
    for i in range(0, size_):
        disp[:, (28*i):(28*(i + 1))] = np.reshape(image[i, :], (28, 28))
    disp = disp*255
    seven = disp.astype(int)
    seven[seven > 255] = 255
    seven[seven < 0] = 0
    plt.imshow(seven, cmap = plt.get_cmap('gray'))
    plt.show()
    plt.imsave(os.getcwd() + "/Test_images/result" + str(np.random.randint(0, 10000)) + ".png", seven, cmap = plt.get_cmap('gray'))