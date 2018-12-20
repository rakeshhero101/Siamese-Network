'''
Advanced Topics in Artificial Intelligence IFN 680
Assignment 2: Siamese Network

Team members:
Mary Rose Legaspi    	N10086820
Minwoo Kang             N9913351
Rakesh Maharjan       	N10032711

'''

from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, MaxPooling2D,Conv2D
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split 
from keras import backend as K
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

#-----------------------------------------------------------------------------#
#                        Load MNIST DATA set and Split it                     #
#-----------------------------------------------------------------------------#
# Loading the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#splitting the dataset 
train_mask = np.isin(y_train, [2, 3, 4, 5, 6, 7])
test_mask = np.isin(y_train, [0,1,8,9])

#digits 2,3,4,5,6,7 for training set and testing set
f_x_train, f_y_train = x_train[train_mask], y_train[train_mask]


#splitting the training into 80% training and 20% evaluation
print("Total training data points: {}".format(len(f_x_train))) # total data for training is 35535
tr_x_train, te_x_val, tr_y_train, te_y_val = train_test_split(f_x_train, f_y_train, test_size=0.20, random_state=42)
print("training data points: {}".format(len(tr_x_train))) # 80% training set 28428
print("validation data points: {}".format(len(te_x_val))) # 20% test set 7107


#digits 0,1,8,9 for testing only from the original training data'
'there are 24465 data'
f_x_test, f_y_test = x_train[test_mask], y_train[test_mask]
print("testing data points: {}".format(len(f_x_test))) 


## Reshaping the array to 4-dims so that it can work with the Keras API
tr_x_train = tr_x_train.reshape(tr_x_train.shape[0], 28, 28, 1)
te_x_val = te_x_val.reshape(te_x_val.shape[0], 28, 28, 1)
f_x_test = f_x_test.reshape(f_x_test.shape[0], 28, 28, 1)
'All digits 0,1,2,3,4,5,6,7,8,9'
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


# Making sure that the values are float so that we can get decimal points after division
tr_x_train = tr_x_train.astype('float32')
te_x_val = te_x_val.astype('float32')
f_x_test = f_x_test.astype('float32')
'All digits 0,1,2,3,4,5,6,7,8,9'
x_train = x_train.astype('float32')


# Normalizing the RGB codes by dividing it to the max RGB value.
tr_x_train /= 255
te_x_val /= 255
f_x_test /= 255
x_train /= 255


print('x_train shape:', tr_x_train.shape)
print('Number of images in x_train', tr_x_train.shape[0])
print('Number of images in x_test', te_x_val.shape[0])


# Identifying the number of classes
num_classes1 = len(np.unique(f_y_train))
num_classes2 = len(np.unique(f_y_test))
num_classesAll=num_classes1+num_classes2
print(tr_x_train.shape[0:])
input_shape = (28, 28, 1)

# group pair for testing the siamese in network using the test set
group1 = np.isin(y_test, [2, 3, 4, 5, 6, 7])
group2 = np.isin(y_test, [0,1,8,9])
pair1_X,pair1_y= x_test[group1], y_test[group1]
pair2_X,pair2_y= x_test[group2], y_test[group2]
unionPair_X,unionPair_y= x_test, y_test

#-----------------------------------------------------------------------------#
#                        constrastive loss function                           #
#-----------------------------------------------------------------------------#
def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

#-----------------------------------------------------------------------------#
#                          Siamese network with CNN                           #
#-----------------------------------------------------------------------------#
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_cnn_network(input_shape):
    '''
    Creating  network
    Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    model = Sequential() (input)
    model=Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1),
                 padding='valid', input_shape=input_shape)  (model)
    model=MaxPooling2D(pool_size=(2, 2)) (model)
    model=Dropout(0.2) (model)
    model=Flatten() (model)
    model=Dense(512, activation='relu') (model)
    model=Dropout(0.2) (model)
    model=Dense(512, activation='relu') (model)
    model=Dropout(0.2) (model)
    model=Dense(512, activation='relu') (model)
    model=Dropout(0.1) (model)
    model=Dense(256, activation='relu') (model)
    model=Dropout(0.1) (model)    
    model=Dense(128, activation='relu') (model)
    model=Dropout(0.1) (model)
    model=Dense(64, activation='relu')(model)
    model=Dropout(0.1)(model)
    model=Dense(num_classesAll, activation='softmax') (model)
    
    return Model(input, model)

def create_pairs(x, digit_indices,num_classes):
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1,0]
    return np.array(pairs), np.array(labels)



def compute_accuracy(y_true, y_pred):
    '''
    Evaluation
    Compute classification accuracy with a fixed threshold on distances.
    
    @ param: pred= the prdiction with threshold
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype))) 



def train(epoch):
    '''
    Training the siamese network using the training set. 
    Create postive and negative pair with the training and test set.
    Create network definition by passing the input shape in the the cnn network
    Processed the shared weight of the two input
    Compute the distance.
    Create a model with two set of input and distance shared.
    Use RMS as the optimizer 
    Compile the model using the contrastive loss function and rms optimizer.
    Fit the model with the training pairs and validate with the test pair.
    Compute the accuracy of the model and plot the  loss and accuracy of both training and vaidation set.
       
    '''
    # create training+test positive and negative pairs
    digit_indices = [np.where(tr_y_train == i)[0] for i in range(2,8)]
    tr_pairs, tr_y = create_pairs(tr_x_train, digit_indices,num_classes1)

    digit_indices = [np.where(te_y_val == i)[0] for i in range(2, 8)]
    te_pairs, te_y = create_pairs(te_x_val, digit_indices,num_classes1)

    # network definition
    base_network = create_cnn_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)


    # Here, the weights of the network  will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    #-----------------------------------------------------------------------------#
    #                         Training siamese network                            #
    #-----------------------------------------------------------------------------#

   
    nb_epoch = epoch
    history=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=512,
                        epochs=nb_epoch,
                        verbose=1,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))


    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]],batch_size=512)
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]], batch_size=512)
    te_acc = compute_accuracy(te_y, y_pred)
    print('')
    print('Results of trained siamese network')
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on validation set: %0.2f%%' % (100 * te_acc))
    
    ' Plot the training and validation error vs time.'
    # Plot the results
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    train_acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    xc=range(nb_epoch)
    
    print('')
    print('The graphs show the loss and accuracy during training and testing of the siamese convolutional neural network')
    plt.subplot(211)
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss and val_loss')
    plt.grid(True)
    plt.legend(['training','validation'])
    plt.style.use(['classic'])
    
    'To save the graph titled "Loss.jpeg"---NOTE YOU MUST CHANGE THE PATH'
    #plt.savefig("C:/Users/Mary/Desktop/Loss.jpeg") # to save the graph
    plt.show()
 
    plt.subplot(212)
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc and val_acc')
    plt.grid(True)
    plt.legend(['training','validation'], loc=4)
    plt.style.use(['classic'])
    
    'To save the graph titled "Accuracy.jpeg"---NOTE YOU MUST CHANGE THE PATH'
    #plt.savefig("C:/Users/Mary/Desktop/Accuracy.jpeg")
    plt.show()

    return model


def pair_eval(mod):

    '''
    Evaluate the simaese network with different group pair.
    This evaluation used the test data of the original mnist dataset.
    The model that is passed into this is from the training (train())
    variables:group1=digits 2,3,4,5,6,7
              group2=digit 0,1,8,9
              unionPair=0,1,2,3,4,5,6,7,8,9
    '''
     #--------testing it with pairs from [2,3,4,5,6,7] x [2,3,4,5,6,7] -----------#

    #test is performed for 5 times with different numbers
    model=mod
    case1= []
    for i in range(5):
       # create pairs
       digit_indices = [np.where(pair1_y == i)[0] for i in np.unique(pair1_y)]
       all_te_pairs, all_y_te = create_pairs(pair1_X, digit_indices,num_classes1)
       
       # predict and print accuracy
       all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
       all_te_acc = compute_accuracy(all_y_te, all_y_pred)
       p1 = ('%0.2f%%' % (100 * all_te_acc))
       case1.append(p1)
       
       #print('* testing with pairs of [2,3,4,5,6,7] x [2,3,4,5,6,7]: %0.2f%%' % (100 * all_te_acc))
    
    
    #--------testing it with pairs from [2,3,4,5,6,7] x [0,1,8,9] ---------------#
    case2= []
    for i in range(5):
        # create pairs
        digit_indices = [np.where(unionPair_y == i)[0] for i in np.unique(unionPair_y)]
        all_te_pairs, all_y_te = create_pairs(unionPair_X, digit_indices,num_classesAll)
    

        # predict and print accuracy
        all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
        all_te_acc = compute_accuracy(all_y_te, all_y_pred)
        p2 = ('%0.2f%%' % (100 * all_te_acc))
        case2.append(p2)
        #print('* testing with pairs of [2,3,4,5,6,7] x [0,1,8,9]: %0.2f%%' % (100 * all_te_acc))

    #-------------testing it with pairs from [0,1,8,9] x [0,1,8,9]----------------#
    case3= []
    for i in range(5):
        # create pairs
        digit_indices = [np.where(pair2_y == i)[0] for i in np.unique(pair2_y)]
        all_te_pairs, all_y_te = create_pairs(pair2_X, digit_indices,num_classes2)

        # predict and print accuracy
        all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
        all_te_acc = compute_accuracy(all_y_te, all_y_pred)
        p3 = ('%0.2f%%' % (100 * all_te_acc))
        case3.append(p3)

    #print('* testing with pairs of [0,1,8,9] x [0,1,8,9]: %0.2f%%' % (100 * all_te_acc))

    #---------------------Results with 5 times test for each----------------------#
    print('------------------------------------------------------------------------')
    print('Test results of each pair with 5 times iteration')
    print('Testing with pairs from [2,3,4,5,6,7] x [2,3,4,5,6,7]')
    print(case1)
    print('')
    print('Testing with pairs from [2,3,4,5,6,7] x [0,1,8,9]')
    print(case2)
    print('')
    print('Testing with pairs from [0,1,8,9] x [0,1,8,9]')
    print(case3)
    print('')
#--------------------------------------------------------------------------------------------------#
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''                                                                              '''
'''                            Important                                         '''
'''                                                                              '''
'''          This is for accumulating results with different epoch values        '''
'''          If you want to show the basic requirements of the project please        '''
'''          do not run this part                                           '''
'''                                                                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def test_eval_with_increasing_epoch(increment):
    '''  
    Train the network in 5 iteration with changing epoch number. The epoch starts at 10 and
    increments by 10 for each iteration.
    Then, create a list to save the acurracy  for the training and evaluation sets. 
    The accuaracy will be displayed in a dataframe format for easy comparison.
    '''
    
    # create training+test positive and negative pairs
    digit_indices = [np.where(tr_y_train == i)[0] for i in range(2,8)]
    tr_pairs, tr_y = create_pairs(tr_x_train, digit_indices,num_classes1)

    digit_indices = [np.where(te_y_val == i)[0] for i in range(2, 8)]
    te_pairs, te_y = create_pairs(te_x_val, digit_indices,num_classes1)

    # network definition
    base_network = create_cnn_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)


    # Here, the weights of the network  will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    cols=["Epochs", "Acc_train_set", "Acc_test_set", "Case_1", "Case_2", "Case_3" ]
    acc=[]
    nb_epoch=0

    for i in range(5):

        nb_epoch += increment
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                                batch_size=512,
                                epochs=nb_epoch,
                                verbose=1,
                                validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    
    
        # compute final accuracy on training and test sets
        y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]],batch_size=512)
        tr_acc = compute_accuracy(tr_y, y_pred)
        y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]], batch_size=512)
        te_acc = compute_accuracy(te_y, y_pred)
        #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        #print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    
        'Testing the model with pairs from [2,3,4,5,6,7] x [2,3,4,5,6,7]'
        # create pairs
        digit_indices = [np.where(pair1_y == i)[0] for i in np.unique(pair1_y)]
        all_te_pairs, all_y_te = create_pairs(pair1_X, digit_indices,num_classes1)
        # predict accuracy
        all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
        t1 = compute_accuracy(all_y_te, all_y_pred)
    
        'Testing the model with pairs from [2,3,4,5,6,7] x [0,1,8,9]'
        # create pairs
        digit_indices = [np.where(unionPair_y == i)[0] for i in np.unique(unionPair_y)]
        all_te_pairs, all_y_te = create_pairs(unionPair_X, digit_indices,num_classesAll)
        # predict accuracy
        all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
        t2 = compute_accuracy(all_y_te, all_y_pred)

        'Testing the model with pairs from [0,1,8,9] x [0,1,8,9]'
        # create pairs
        digit_indices = [np.where(pair2_y == i)[0] for i in np.unique(pair2_y)]
        all_te_pairs, all_y_te = create_pairs(pair2_X, digit_indices,num_classes2)
        # predict accuracy
        all_y_pred = model.predict([all_te_pairs[:, 0], all_te_pairs[:, 1]], verbose=True,batch_size=512)
        t3 = compute_accuracy(all_y_te, all_y_pred)

        # Append the results of each iteration in a list and create a dataframe to display result  
        acc.append([nb_epoch,"%.2f" %(tr_acc*100), "%.2f" %(te_acc*100), "%.2f" %(t1*100),"%.2f" %(t2*100),"%.2f" %(t3*100)])
        df1 = pd.DataFrame(acc, columns=cols)
    
    # to save the dataframe in CSV format-NOTE YOU MUST CHANGE THE PATH
    #df1.to_csv("C:/Users/Mary/Desktop/results.csv", index=False, header=True)
    
    
    #Print the dataframe containing the accuracy results 
    print('------------------------------------------------------------------------')
    print('The table shows the results retrieved from different epoch values')
    print('')
    print(df1)


    
#========================================================================================================#    
if __name__ == '__main__':
     
    model=train(epoch=20)# here we use epoch 10,20 and 30 / for debugging we use lower values
    pair_eval(model)
#   test_eval_with_increasing_epoch(increment=10) # for the evaluation we use 10
                                                 # for debugging we use smaller increment size
    