import pandas as pd
import numpy as np
import data_handler as dh
from lstms import LSTMModel
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

df_train, df_test, scaler, train, test = dh.create_data(kind='+', output=3, periods_out=6)

# MODEL
# Number of features
input_dim = df_train.shape[-1]-1
hidden_dim = 100
layer_dim = 2
batch_size = 1
n_steps = 15
# Number of outputs
output_dim = 1



model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()


def train(n_epochs, df_train, df_test):
    best_val = 1000
    train_loss = []
    for epoch in range(n_epochs):
        for batch in range(df_train.shape[0]-n_steps-1):
            x_batch, y_batch = dh.next_stock_batch(batch_size, df_train, n_steps, np.array([batch]))
            x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
            x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(y_batch[:, -1].flatten(), outputs.flatten())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        for batch_test in range(df_test.shape[0]-n_steps-1):
            x_batch, y_batch = dh.next_stock_batch(batch_size, df_test, n_steps, np.array([batch_test]))
            x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
            x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()
            with torch.no_grad():
                outputs = model(x_batch)
                test_loss = criterion(y_batch[:, -1].flatten(), outputs.flatten())
            if best_val > test_loss:
                best_val = test_loss
                torch.save(model, f"Model_ldim{layer_dim}_nsteps{n_steps}.pth")
        print(epoch, "\t Train Loss:", loss.item(), "\t Test Loss:", test_loss.item())

    plt.plot(train_loss, label="Train Loss")
    plt.xlabel(" Iteration ")
    plt.ylabel("Loss value")
    plt.legend(loc="upper left")
    plt.show()
    # plt.clf()


train(5, df_train, df_test)


def test():
    LSTM = torch.load(f"Model_ldim{layer_dim}_nsteps{n_steps}.pth")

    x_batch, y_batch = dh.next_stock_batch(batch_size, df_train, n_steps)
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
    x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()

    with torch.no_grad():
        LSTM.eval()
        outputs = LSTM(x_batch)
        loss = criterion(y_batch[:, -1].flatten(), outputs.flatten())
        print(loss)

    y = y_batch.numpy().reshape((batch_size, input_dim))[:, -1]
    o = outputs.numpy().reshape((batch_size, output_dim)).flatten()

    # plt.plot(y, label="Ground truth")
    # plt.plot(o, label="Prediction")
    # plt.xlabel(" Time ")
    # plt.ylabel("RV return")
    # plt.legend(loc="upper left")
    # # plt.savefig('seq1.png')
    # plt.show()
    # # plt.clf()

    x_batch, y_batch = dh.next_stock_batch(batch_size, df_test, n_steps)
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
    x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()

    with torch.no_grad():
        LSTM.eval()
        outputs = LSTM(x_batch)
        loss = criterion(y_batch[:, -1].flatten(), outputs.flatten())
        print(loss)

    y = y_batch.numpy().reshape((batch_size, input_dim))[:, -1]
    o = outputs.numpy().reshape((batch_size, output_dim)).flatten()

    # Trick to unscale the data
    y_copies = np.repeat(y, input_dim, axis=-1).reshape(batch_size, -1)
    o_copies = np.repeat(o, input_dim, axis=-1).reshape(batch_size, -1)
    y = scaler.inverse_transform(y_copies)[:, 0]
    o = scaler.inverse_transform(o_copies)[:, 0]

    # plt.plot(y, label="Ground truth")
    # plt.plot(o, label="Prediction")
    # plt.xlabel(" Time ")
    # plt.ylabel("RV return")
    # plt.legend(loc="upper left")
    # # plt.savefig(f'Seq{period}_ldim{layer_dim}_nsteps{n_steps}.png')
    # plt.show()
    # # plt.clf()
