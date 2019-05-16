import tensorflow as tf
import numpy as np
import os as os
import timeit

'''
Class backprop:
---------------
Constructor Parameters:
    1. input: training input to the model (numpy array)
    2. target: target values for supervised training (numpy array)
    3. n_hidden: number of hidden layers (int)
    4. hidden: number of neurons in each layer (list of length equal to n_hidden) 
    5. max_iter: maximum number of iterations to be run (int)
    6. eta: learning rate (float)
    7. f_type: forward neural network type (regression or classification) 
            (string: Only accepted values are "R" : regression and "C" : classification)
    8. b_type: reverse neural network type (regression or classification)
            (string: Only accepted values are "R" : regression and "C" : classification)
    9. b: batch size (int)
    10. disp_step: progress display variable (int)
'''

class backprop():
    def __init__(self, input, target, n_hidden, hidden, max_iter = 10000, eta = 1, f_type = "R", b_type = "R", b = 100, disp_step = 10):
        self.__input = input
        self.__target = target
        self.__n_hidden = n_hidden
        self.__hidden = hidden
        self.__n = max_iter
        self.__eta = eta
        self.__f_type = f_type
        self.__b_type = b_type
        self.__b = b
        self.__disp_step = disp_step
        with tf.device("/gpu:0"):
            self.__w = [tf.Variable(tf.random_normal([input.shape[1], hidden[0]]), name = 'weight_in')]
            for i in range(1, n_hidden):
                self.__w.append(tf.Variable(tf.random_normal([hidden[i - 1], hidden[i]]), name = "weights_h_" + str(i)))
            self.__w.append(tf.Variable(tf.random_normal([hidden[-1], target.shape[1]]), name = "weight_out"))
            self.__b_f = [tf.Variable(tf.random_normal([1, hidden[0]]), name = "bias_f_in")]
            for i in range(1, n_hidden):
                self.__b_f.append(tf.Variable(tf.random_normal([1, hidden[i]]), name = "bias_f_h_" + str(i)))
            self.__b_f.append(tf.Variable(tf.random_normal([1, target.shape[1]]), name = "bias_f_out"))
            self.__b_b = [tf.Variable(tf.random_normal([1, hidden[-1]]), name = "bias_b_in")]
            for i in range(n_hidden - 2, -1, -1):
                self.__b_b.append(tf.Variable(tf.random_normal([1, hidden[i]]), name = "bias_b_h_" + str(i)))
            self.__b_b.append(tf.Variable(tf.random_normal([1, input.shape[1]]), name = "bias_b_out"))
        self.__sub_dec = np.random.randint(10000)
        self.__X = tf.placeholder(tf.float32, [None, input.shape[1]], name = "Input")
        self.__Y = tf.placeholder(tf.float32, [None, target.shape[1]], name = "Target")
    
    '''
    Function to define the activation function to use:
    Traditionally, regression networks use the sigmoid activation function,
    and the classification models use the softmax activation function for 
    the output layer. To simplify the code, I have used these activation functions
    throughout the network. Future implementations will increase flexibility

    Arguments: 
        type: Type of the network used.
    
    Returns the activation function to be used
    '''

    def __act_func(self, type = "R"):
        if type == "R":
            return tf.nn.sigmoid
        elif type == "C":
            return tf.nn.softmax
        else:
            raise ValueError("Only regression and classification networks are supported!")

    '''
    Function to generate the forward network:
    This network generates the forward network using the weights defined earlier.

    Arguments:
        None
    
    Returns the output of the forward network
    '''

    def __forward_builder(self):
        hidden_f = []
        act = self.__act_func(self.__f_type)
        hidden_f.append(tf.add(tf.matmul(self.__X, self.__w[0]), self.__b_f[0]))
        hidden_f[0] = act(hidden_f[0])
        for i in range(1, self.__n_hidden):
            hidden_f.append(tf.add(tf.matmul(hidden_f[i - 1], self.__w[i]), self.__b_f[i]))
            hidden_f[i] = act(hidden_f[i])
        out_f = tf.add(tf.matmul(hidden_f[-1], self.__w[-1]), self.__b_f[-1])
        return out_f

    '''
    Function to generate the reverse network:
    This network generates the reverse network using the weights defined earlier.

    Arguments:
        None
    
    Returns the output of the reverse network
    '''
    def __reverse_builder(self):
        hidden_b = []
        act = self.__act_func(self.__b_type)
        hidden_b.append(tf.add(tf.matmul(self.__Y, tf.transpose(self.__w[-1])), self.__b_b[0]))
        hidden_b[0] = act(hidden_b[0])
        for i in range(1, self.__n_hidden):
            hidden_b.append(tf.add(tf.matmul(hidden_b[i - 1], tf.transpose(self.__w[self.__n_hidden - i])), self.__b_b[i]))
            hidden_b[i] = act(hidden_b[i])
        out_b = tf.add(tf.matmul(hidden_b[-1], tf.transpose(self.__w[0])), self.__b_b[-1])
        return out_b

    '''
    Function to generate batches

    Arguments:
        index: the array of indices based on the input
    
    Returns the batches made from the index array
    '''
    
    def __index_shuffler(self, index):
        n = self.__input.shape[0]
        np.random.shuffle(index)
        n_batches = int(n/self.__b)
        j = np.random.randint(0, n_batches)
        return index[j*self.__b:(j + 1)*self.__b]
    
    '''
    Function to calculate the centroids of the input space:
    Given an batch of inputs and a batch of targets, 
    generates the centroids for a classifcation problem.

    Arguments:
        batch_x: input batch
        batch_y: target batch
    
    Returns the centroids for each class
    '''
    
    def __centroid (self, batch_x, batch_y):
        num = batch_x.shape[0]
    
        batch_z = np.zeros([num, self.__input.shape[1]])
        tag = np.argmax(batch_y, 1)
        
        for it in range(self.__target.shape[1]):
            
            idx = (tag == it)
            batch_z[idx, :] = np.mean(batch_x[idx,:], 0)
        
        return batch_z  
    
    '''
    Function to build a dictionary of weights to be saved:
    Arguments:
        None
    
    Returns the model as a dictionary
    '''

    def __dict_builder(self):
        saved_model = { 'weight_in' : self.__w[0]}
        for i in range(1, self.__n_hidden):
            saved_model['weights_h_' + str(i - 1)] = self.__w[i]
        saved_model['weight_out'] = self.__w[self.__n_hidden]
        saved_model['bias_f_in'] = self.__b_f[0]
        for i in range(1, self.__n_hidden):
            saved_model['bias_f_h_' + str(i - 1)] = self.__b_f[i]
        saved_model['bias_f_out'] = self.__b_f[self.__n_hidden]
        saved_model['bias_b_in'] = self.__b_b[0]
        for i in range(1, self.__n_hidden):
            saved_model['bias_b_h_' + str(i - 1)] = self.__b_b[i]
        saved_model['bias_b_out'] = self.__b_b[self.__n_hidden]
        return saved_model
    
    '''
    Function to train the model:
    Given the input and the targets in the constructor call,
    this function trains the network in both the forward and the reverse directions.
    It saves the model, and generates Tensorboard objects for the loss, accuracy,
    weights, biases etc.
    It then displays the error every disp_step steps (in case of classification, it
    also displays the accuracy)

    Argumnets:
        None
    
    Returns:
        None
    '''
    def __err_correction(self):
        n_batches = int(self.__input.shape[0]/self.__b)
        with tf.device('/gpu:0'):
            out_f = self.__forward_builder()
            if self.__f_type == "R":
                err_f = tf.reduce_mean(tf.losses.mean_squared_error(self.__Y, out_f))
            elif self.__f_type == "C":
                err_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.__Y, logits = out_f))
            else:
                raise ValueError("Only regression and classification networks are supported!")
            optimizer_f = tf.train.AdamOptimizer(self.__eta)
            train_f = optimizer_f.minimize(err_f)

            if (self.__f_type == "C"):
                acc = tf.nn.softmax(out_f)
                acc = tf.equal(tf.argmax(out_f, 1), tf.argmax(self.__Y, 1))
                acc_f = tf.reduce_mean(tf.cast(acc, tf.float32))

            out_b = self.__reverse_builder()
            if self.__b_type =="R":
                err_b = tf.reduce_mean(tf.losses.mean_squared_error(self.__X, out_b))
            elif self.__b_type == "C":
                err_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.__X, logits = out_b))
            else:
                raise ValueError("Only regression and classification networks are supported!")
            optimizer_b = tf.train.AdamOptimizer(self.__eta)
            train_b = optimizer_b.minimize(err_b)
        model_save = self.__dict_builder()
        saver = tf.train.Saver(model_save)
        # weights = tf.stack(self.__w[1:-1])
        # biases_for = tf.stack(self.__b_f[:-1])
        # biases_back = tf.stack(self.__b_b[:-1])
        # tf.summary.histogram("weights", weights)
        # tf.summary.histogram("biases_for", biases_for)
        # tf.summary.histogram("biases_back", biases_back)
        tf.summary.scalar("Forward_Error", err_f)
        tf.summary.scalar("Backward_Error", err_b)
        tf.summary.scalar("Accuracy", acc_f)
        merger = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.getcwd() + "/Tensorboards/Tester_" + str(self.__sub_dec) + "/Bidirec_Backprop")
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        writer.add_graph(sess.graph)
        index_ = np.arange(0, self.__input.shape[0])
        for i in range(0, self.__n):
            avg_err_f = 0
            avg_err_b = 0
            start_time = timeit.default_timer()
            for j in range(0, n_batches):
                index = self.__index_shuffler(index_)
                _, err_for = sess.run([train_f, err_f], feed_dict = {self.__X : self.__input[index, :], self.__Y : self.__target[index, :]})
                _, err_bac = sess.run([train_b, err_b], feed_dict = 
                {self.__X : self.__centroid(self.__input[index, :], self.__target[index, :]), self.__Y : self.__target[index, :]})
                avg_err_f += err_for/n_batches
                avg_err_b += err_bac/n_batches
                s = sess.run(merger, feed_dict = {self.__X : self.__input[index, :], self.__Y : self.__target[index, :]})
                writer.add_summary(s, n_batches*i + j)
            elapsed_time = timeit.default_timer() - start_time
            if (i + 1) % self.__disp_step == 0:
                print("For Epoch " + str(i + 1) + " the average cost is " + str(avg_err_f) + " in the forward direction and " + str(avg_err_b) + " in the reverse direction.")
                print("Time taken: " + str(elapsed_time))
                if(self.__f_type == "C"):
                    print("Accuracy in forward direction is " + str(acc_f.eval({self.__X : self.__input, self.__Y : self.__target}, session = sess)))
        print("Optimization Finished!!")
        save_path = saver.save(sess, os.getcwd() + "/model/model.ckpt")
        print(save_path)
        sess.close()
    
    '''
    Function to train the baseline model:
    Given the input and the targets in the constructor call,
    this function trains the network in only the forward direcion (vanilla backprop)
    It saves the model, and generates Tensorboard objects for the loss, accuracy,
    weights, biases etc.
    It then displays the error every disp_step steps (in case of classification, it
    also displays the accuracy)

    Argumnets:
        test_input: The test set
        test_labels: The test set targets
    
    Returns:
        None
    '''
    
    def baseline(self, test_input, test_labels):
        with tf.device('/gpu:0'):
            X = self.__forward_builder()
            if self.__f_type == "R":
                err_f = tf.reduce_mean(tf.losses.mean_squared_error(self.__Y, X))
                optimizer_base = tf.train.AdamOptimizer(self.__eta)
                train_base = optimizer_base.minimize(err_f)
            else:
                err_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.__Y, logits = X))
                acc_f = tf.nn.softmax(X)
                acc_f = tf.equal(tf.argmax(X, 1), tf.argmax(self.__Y, 1))
                acc = tf.reduce_mean(tf.cast(acc_f, tf.float32))
                optimizer_base = tf.train.AdamOptimizer(self.__eta)
                train_base = optimizer_base.minimize(err_f)
                tf.summary.scalar("Accuracy", acc)

        # weights = tf.stack(self.__w[1:-1])
        # biases_for = tf.stack(self.__b_f[:-1])
        # biases_back = tf.stack(self.__b_b[:-1])
        # tf.summary.histogram("weights", weights)
        # tf.summary.histogram("biases_for", biases_for)
        tf.summary.scalar("Forward_Error", err_f)
        merger = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.getcwd() + "/Tensorboards/Tester_" + str(self.__sub_dec) + "/Baseline")
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        writer.add_graph(sess.graph)
        index_ = np.arange(0, self.__input.shape[0])
        n_batches = int(self.__input.shape[0]/self.__b)
        for i in range(self.__n):
            avg_err = 0
            avg_acc = 0
            for j in range(n_batches):
                index = self.__index_shuffler(index_)
                if self.__f_type == "R":
                    _, err_ = sess.run([train_base, err_f], feed_dict = { self.__X : self.__input[index, :], self.__Y : self.__target[index, :]})
                    avg_err += err_/n_batches
                else:
                    _, err_, acc_f = sess.run([train_base, err_f, acc], feed_dict = { self.__X : self.__input[index, :], self.__Y : self.__target[index, :]})
                    avg_err += err_/n_batches
                    avg_acc += acc_f/n_batches
                s = sess.run(merger, feed_dict = { self.__X : self.__input[index, :], self.__Y : self.__target[index, :]})
                writer.add_summary(s, n_batches*i + j)
            if((i + 1) % self.__disp_step == 0):
                if(self.__f_type == "C"):
                    print("Baseline performance after epoch " + str(i + 1) + " is: Error: " + str(avg_err) + " Accuracy: " + str(avg_acc))
                else:
                    print("Baseline performance after epoch " + str(i + 1) + " is: Error: " + str(avg_err))
        if(self.__f_type == "C"):
            err_test, acc_test = sess.run([err_f, acc], feed_dict = { self.__X : test_input, self.__Y : test_labels})
            print("Baseline performance on testing data: Error: " + str(err_test) + " Accuracy: " + str(acc_test))
        else:
            err_test = sess.run(err_f, feed_dict = { self.__X : test_input, self.__Y : test_labels})
            print("Baseline performance on testing data: Error: " + str(err_test))
        print("Baseline trained!")
        sess.close()
    
    '''
    Function to initialize uninitialized variables (if any):

    Arguments:
        sess: Session under consideration
    
    Returns:
        None
    '''
    
    def __initialize_uninitialized(self, sess):
        global_vars          = tf.global_variables()
        is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))
    
    '''
    Standard score function:

    Arguments:
        input: Input to the network
        target: Target to the network
    
    Returns the forward ans reverse error
    '''

    def score(self, input, target):
        X = self.__forward_builder()
        Y = self.__reverse_builder()
        saved_dict = self.__dict_builder()
        saver = tf.train.Saver(saved_dict)
        if self.__f_type == "R":
            err_f = tf.reduce_mean(tf.losses.mean_squared_error(self.__Y, X))
        elif self.__f_type == "C":
            err_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.__Y, logits = X))
            acc_f = tf.nn.softmax(X)
            acc_f = tf.equal(tf.argmax(X, 1), tf.argmax(self.__Y, 1))
            acc_f = tf.reduce_mean(tf.cast(acc_f, tf.float32))
        else:
            raise ValueError("Only regression and classification networks are supported!")
        if self.__b_type == "R":
            err_b = tf.reduce_mean(tf.losses.mean_squared_error(self.__X, Y))
        elif self.__b_type == "C":
            err_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.__X, logits = Y))
            acc_b = tf.nn.softmax(X)
            acc_b = tf.equal(tf.argmax(Y, 1), tf.argmax(self.__X, 1))
            acc_b = tf.reduce_mean(tf.cast(acc, tf.float32))
        else:
            raise ValueError("Only regression and classification networks are supported!")
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, os.getcwd() + "/model/model.ckpt")
        self.__initialize_uninitialized(sess)
        if(self.__f_type == "C"):
            forward_err = sess.run(acc_f, feed_dict = {self.__X : input, self.__Y : target})
        else:
            forward_err = sess.run(err_f, feed_dict = {self.__X : input, self.__Y : target})
        if(self.__b_type == "C"):
            reverse_err = sess.run(acc_b, feed_dict = {self.__X : input, self.__Y : target})
        else:
            reverse_err = sess.run(err_b, feed_dict = {self.__X : input, self.__Y : target})
        sess.close()
        return forward_err, reverse_err
        
    '''
    Standard predict function:

    Arguments:
        input: input to the network
        direction: The direction in which the input is to be given
    
    Returns:
        The output predictions
    '''
    def predict(self, input, direction = "F"):
        saved_dict = self.__dict_builder()
        saver = tf.train.Saver(saved_dict)
        X = self.__forward_builder()
        Y = self.__reverse_builder()
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, os.getcwd() + "/model/model.ckpt")
        self.__initialize_uninitialized(sess)
        if (direction == "F"):
            pred = sess.run(X, feed_dict = { self.__X : input})
            sess.close()
            return pred
        else:
            pred = sess.run(Y, feed_dict = { self.__Y : input})
            sess.close()
            return pred
        
    '''
    Standard fit function:

    Arguments:
        None. Takes the input and output from the class variables
    
    Returns:
        Errors using the bidirectional backprop algorithm 
        Error using the baseline backpropagation algorithm
    '''
    
    def fit(self):
        self.__err_correction()
        err = self.score(self.__input, self.__target)
        pred = self.predict(self.__input, direction = "F")
        if(self.__f_type == "C"):
            acc = tf.nn.softmax(pred)
            acc = tf.equal(tf.argmax(pred, 1), tf.argmax(self.__Y, 1))
            err_base = tf.reduce_mean(tf.cast(acc, tf.int32))
        else:
            err_base = tf.losses.mean_squared_error(self.__Y, pred)
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        err_ = sess.run(err_base, feed_dict = { self.__X : self.__input, self.__Y : self.__target})
        sess.close()
        return err, err_