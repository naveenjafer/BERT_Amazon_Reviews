import pandas as pd
import torch
import torch.nn as nn
from transformers import  BertModel, BertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import dataLoader
import classifier
import sys

def read_and_shuffle(file):
    df = pd.read_csv(file, delimiter=',')
    # Random shuffle.
    df.sample(frac=1)
    return df

def get_train_and_val_split(df, splitRatio=0.8):
    train=df.sample(frac=splitRatio,random_state=201)
    val=df.drop(train.index)
    print("Number of Training Samples: ", len(train))
    print("Number of Validation Samples: ", len(val))
    return(train, val)

def get_max_length(reviews):
    return len(max(reviews, key=len))

def get_accuracy(logits, labels):
    # get the index of the max value in the row.
    predictedClass = logits.max(dim = 1)[1]

    # get accuracy by averaging over entire batch.
    acc = (predictedClass == labels).float().mean()
    return acc

def trainFunc(net, loss_func, opti, train_loader, test_loader, config):
    best_acc = 0
    for ep in range(config["epochs"]):
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            opti.zero_grad()
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            logits = net(seq, attn_masks)
            loss = loss_func(m(logits), labels)

            loss.backward()
            opti.step()
            print("Iteration: ", it+1)

            if (it + 1) % config["printEvery"] == 0:
                acc = get_accuracy(m(logits), labels)
                if not os.path.exists(config["outputFolder"]):
                    os.makedirs(config["outputFolder"])

                # Since a single epoch could take well over hours, we regularly save the model even during evaluation of training accuracy.
                torch.save(net.state_dict(), os.path.join(config["outputFolder"], config["outputFileName"]))
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))

        # perform validation at the end of an epoch.
        val_acc, val_loss = evaluate(net, loss_func, val_loader, config)
        print(" Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(config["outputFolder"], config["outputFileName"] + "_valTested_" + str(best_acc)))

def saveEvalAndTrain(train, eval, config):
    val.to_csv("AMAZON-DATASET/Validations.csv")
    train.to_csv("AMAZON-DATASET/Train.csv")

def evaluate(net, loss_func, dataloader, config):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            logits = net(seq, attn_masks)
            mean_loss += loss_func(m(logits), labels)
            mean_acc += get_accuracy(m(logits), labels)
            print("Validating reviews ", count * config["batchSize"], " - ", (count+1) * config["batchSize"])
            count += 1

            '''
            The entire validation set was around 0.1 million entries,
            the validationFraction param controls what fraction of the shuffled
            validation set you want to validate the results on.
            '''
            if count > config["validationFraction"] * len(val_set):
                break

    return mean_acc / count, mean_loss / count

if __name__== "__main__":
    arguments = sys.argv[1:]
    trainOrEval = arguments[0]

    config = {
    "splitRatio" : 0.8,
    "maxLength" : 100,
    "printEvery" : 5,
    "outputFolder" : "Models",
    "outputFileName" : "AmazonReviewClassifier.dat",
    "threads" : 4,
    "batchSize" : 64,
    "validationFraction" : 0.0005,
    "epochs" : 5,
    "forceCPU" : False
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["forceCPU"]:
        device = torch.device("cpu")

    config["device"] = device

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print("Configuration is: ", config)
    # Read and shuffle input data.
    df = read_and_shuffle("./AMAZON-DATASET/Reviews.csv")

    num_classes = df['Score'].nunique()
    config["num_classes"] = num_classes
    print("Number of Target Output Classes:", num_classes)
    totalDatasetSize = len(df)

    # Group by the column Score. This helps you get distribution of the Review Scores.
    symbols = df.groupby('Score')

    scores_dist = []
    for i in range(num_classes):
        scores_dist.append(len(symbols.groups[i+1])/totalDatasetSize)

    train, val = get_train_and_val_split(df, config["splitRatio"])

    if trainOrEval == "eval":
        try:
            val = pd.read_csv("./AMAZON-DATASET/Validations.csv", delimiter=',')
        except:
            print("Could not find file Validations.csv in AMAZON-DATSET, run the training and then run the eval flag.")
    if trainOrEval == "train":
        saveEvalAndTrain(train, eval, config)
    # You can set the length to the true max length from the dataset, I have reduced it for the sake of memory and quicker training.
    #T = get_max_length(reviews)
    T = config["maxLength"]

    train_set = dataLoader.AmazonReviewsDataset(train, T)
    val_set = dataLoader.AmazonReviewsDataset(val, T)

    '''
    Uncomment this block if you would like to see what is contained in token_ids, attention_masks and labels.
    _1, _2, _3 = train_set.__getitem__(0)
    print(_1,_2,_3)
    '''

    train_loader = DataLoader(train_set, batch_size = config["batchSize"], num_workers = config["threads"])
    val_loader = DataLoader(val_set, batch_size = config["batchSize"], num_workers = config["threads"])

    # We are unfreezing the BERT layers so as to be able to fine tune and save a new BERT model that is specific to the Sizeable food reviews dataset.
    net = classifier.SentimentClassifier(num_classes, config["device"], freeze_bert=False)
    net.to(config["device"])
    weights = torch.tensor(scores_dist).to(config["device"])

    # Setting the Loss function and Optimizer.
    loss_func = nn.NLLLoss(weight=weights)
    opti = optim.Adam(net.parameters(), lr = 2e-5)

    m = nn.LogSoftmax(dim=1)

    if trainOrEval == "train":
        trainFunc(net, loss_func, opti, train_loader, val_loader, config)

    elif trainOrEval == "eval":
        model = classifier.SentimentClassifier(5, device, freeze_bert = False)
        try:
            model.load_state_dict(torch.load(os.path.join(config["outputFolder"], config["outputFileName"]), map_location=config["device"]))
            print("Loaded model at ", os.path.join(config["outputFolder"], config["outputFileName"]))
        except:
            print("Failed to load the model weights, please check if the file ", os.path.join(config["outputFolder"], config["outputFileName"]), "exisits and is not corrupt")
        val_acc, val_loss = evaluate(model, loss_func, val_loader, config)
        print("Accuracy: ",val_acc,"\nLoss: ", val_loss)

    else:
        print("Please give a command line argument of either 'train' or 'eval'. For example. python3 main.py eval\n Eval will only work once the first model from training is available.")
