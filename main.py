import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
np.random.seed(100)
torch.manual_seed(100)
seed = 100

# =================================== LOAD DATASET =========================================== #

######

# 2.1 YOUR CODE HERE
data = pd.read_csv("./data/adult.csv")

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 2.2 YOUR CODE HERE
# The 'shape' is the dimensions of the table
print("Shape: ", data.shape)

# Print the names of the columns - comes from first line of spreadsheet
print(data.columns)

# Print the first 5 rows
verbose_print(data.head())

# print out the entire column - see can reference by the column head name
print(data["income"])
print(data["income"].value_counts())


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

    # 2.3 YOUR CODE HERE

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    print("", feature, data[feature].isin(["?"]).sum())


# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error


    # 2.3 YOUR CODE HERE
data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]
print(data.shape)
# There are 45222 rows remaining.


# =================================== BALANCE DATASET =========================================== #


    # 2.4 YOUR CODE HERE
print(data["income"].value_counts())              # smallest set = >50K with 11208 examples (vs. 34014)

data_rand = data[data["income"] == "<=50K"].sample(n=11208,random_state=seed)       #randomly sample <=50K dataset
data = data_rand.append(data[data["income"] == ">50K"])                             #append the two
data = data.sample(frac=1)                                                          #shuffle the rows

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

    # 2.5 YOUR CODE HERE
verbose_print(data.describe())


# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs

    # 2.5 YOUR CODE HERE
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    print("feature: ", feature)
    print(data[feature].value_counts())

# visualize the first 3 features using pie and bar graphs

# 2.5 YOUR CODE HERE
for i in range(3):
    binary_bar_chart(data, categorical_feats[i])
    pie_chart(data, categorical_feats[i])



# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
# 2.6 YOUR CODE HERE
continuous_feats = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cont = data[continuous_feats]
cont_norm = ((cont - cont.mean())/cont.std())
data_cont = cont_norm.values

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()

# 2.6 YOUR CODE HERE
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

cat = data[categorical_feats]
data_cat = pd.DataFrame()

for feature in categorical_feats:
    data_cat[feature] = label_encoder.fit_transform(cat[feature].values)

oneh_encoder = OneHotEncoder()

# 2.6 YOUR CODE HERE
income = data_cat["income"].values
data_cat = data_cat.drop(columns=["income"])                # remove income for one hot encoding
data_cat_oneh = oneh_encoder.fit_transform(data_cat.values).toarray()       # onehot code

data_clean = np.concatenate([data_cont, data_cat_oneh], axis=1)

# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

# 2.7 YOUR CODE HERE
feat_train, feat_valid, label_train, label_valid = train_test_split(data_clean, income, test_size=0.2, random_state=seed)


# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    # 3.2 YOUR CODE HERE
    train_dataset = AdultDataset(feat_train, label_train)
    valid_dataset = AdultDataset(feat_valid, label_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr):
    # 3.4 YOUR CODE HERE
    loss_fnc = torch.nn.BCELoss()
    model = MultiLayerPerceptron(feat_train.shape[1])
    optimizer = torch.optim.adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    # 3.6 YOUR CODE HERE
    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        feats = feats.float()
        prediction = model.forward(feats)
        corr = (prediction > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())
    return float(total_corr) / len(val_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    MaxEpochs = args.epochs
    lr = args.lr
    batchsize = args.batch_size
    eval_every = args.eval_every

    model, loss_fnc, optimizer = load_model(lr=lr)
    train_loader, val_loader = load_data(batchsize)
    valid_acc_array = []
    train_array = []

    t = 0
    t_array = []
    tot_corr_sum = 0
    t_0 = time()
    time_array = []

    # 3.5 YOUR CODE HERE
    for epoch in range(MaxEpochs):
        accum_loss = 0
        tot_corr = 0

        for i, batch in enumerate(train_loader):
            # this gets one "batch" of data
            feats, label = batch  # feats will have shape (batch_size,4)

            # need to send batch through model and do a gradient opt step;
            # first set all gradients to zero
            optimizer.zero_grad()

            # Run the neural network model on the batch, and get answers
            feats = feats.float()
            predictions = model(feats)  # has shape (batch_size,1)

            # compute the loss function (BCE as above) using the correct answer for the entire batch
            # label was an int, needs to become a float
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            accum_loss += batch_loss

            # computes the gradient of loss with respect to the parameters
            # to make this possible;  uses back-propagation
            batch_loss.backward()

            # Change the parameters  in the model with one 'step' guided by the learning rate.
            # Recall parameters are the weights & bias
            optimizer.step()

            # calculate number of correct predictions
            corr = (predictions > 0.5).squeeze().long() == label
            tot_corr += int(corr.sum())
            tot_corr_sum += int(corr.sum())


            # evaluate model on the validation set every eval_every steps
            if (i+1) % args.eval_every == 0:
                valid_acc = evaluate(model, val_loader)
                print("Epoch: {}, Step {} | Total Correct: {}| Test acc: {}".format(epoch+1, i, tot_corr, valid_acc))
                accum_loss = 0
                train_acc = tot_corr_sum/(len(label)*eval_every)
                tot_corr_sum = 0
                t_array.append(t)
                time_array.append(time()-t_0)

                train_array.append(train_acc)
                valid_acc_array.append(valid_acc)
            t = t+1

    print("training loop time: ", time()-t_0)
    print("bs: ", batchsize, "max val acc:", max(valid_acc_array))
    print(max(train_array))


    # Plot Validation and Training Data
    plt.figure()
    plt.title("Validation and Training Accuracy Over Number of Gradient Steps")
    plt.plot(t_array, train_array, label="Training")
    plt.plot(t_array, valid_acc_array, label="Validation")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("ValandTrain_LAST.png")

    # Plot with smooth training data
    from scipy.signal import savgol_filter
    plt.figure()
    plt.title("Smoothed training data")
    plt.plot(t_array, savgol_filter(train_array, 3, 1), label="Training")
    plt.plot(t_array, valid_acc_array, label="Validation")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("Smooth_ValandTrain_LAST.png")

    # Plot against time
    # plt.figure()
    # plt.title("Batch Size = 17932 (Against Time)")
    # plt.plot(time_array, train_array, label="Training")
    # plt.plot(time_array, valid_acc_array, label="Validation")
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best')
    # plt.savefig("ValandTrain_bs_17932_time.png")


    # Plot all three activation functions
    # tanh = False
    # relu = False
    # sigmoid = True
    #
    # if tanh:
    #     np.savetxt("tanh_train.csv", train_array)
    #     np.savetxt("tanh_valid.csv", valid_acc_array)
    # if relu:
    #     np.savetxt("relu_train.csv", train_array)
    #     np.savetxt("relu_valid.csv", valid_acc_array)
    # if sigmoid:
    #     np.savetxt("sigmoid_train.csv", train_array)
    #     np.savetxt("sigmoid_valid.csv", valid_acc_array)
    #
    #
    # [tanh_train, tanh_valid] = [np.genfromtxt('tanh_train.csv'), np.genfromtxt('tanh_valid.csv')]
    # [relu_train, relu_valid] = [np.genfromtxt('relu_train.csv'), np.genfromtxt('relu_valid.csv')]
    # [sigmoid_train, sigmoid_valid] = [np.genfromtxt('sigmoid_train.csv'), np.genfromtxt('sigmoid_valid.csv')]
    #
    # plt.figure()
    # plt.title("Training and Validation of Different Activation Functions")
    # plt.plot(t_array, tanh_train, label="Training (Tanh)")
    # plt.plot(t_array, tanh_valid, label="Validation (Tanh)")
    # plt.plot(t_array, relu_train, label="Training (ReLU)")
    # plt.plot(t_array, relu_valid, label="Validation (ReLU)")
    # plt.plot(t_array, sigmoid_train, label="Training (Sigmoid)")
    # plt.plot(t_array, sigmoid_valid, label="Validation (Sigmoid)")
    # plt.xlabel("Number of Gradient Steps")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best')
    # plt.savefig("ActivationFunctions.png")


if __name__ == "__main__":
    main()