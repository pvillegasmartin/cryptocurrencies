import data_handler as dh
from lstms import Model
from torch.optim import Adam, SGD
from torch.autograd  import Variable
from torch import nn
import torch
import matplotlib.pyplot as plt


df_train, df_test = dh.create_data()


n_steps = 40 
n_inputs = len(df_train.columns)
n_neurons = 1
hidden_dim = n_steps
n_outputs = 1
learning_rate = 0.01
batch_size = 64

LSTM = Model(n_inputs, n_neurons, hidden_dim)

LSTM.cuda()

optim = Adam(LSTM.parameters(), lr= learning_rate)

criterion = nn.L1Loss()


def train(n_iterations, df_train, df_test):

    best_val = 1000
    train_loss = []
    for iter in range(n_iterations):

        x_batch, y_batch = dh.next_stock_batch(batch_size, hidden_dim, df_train, df_test)
        x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
        x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()

        x_batch.cuda()
        y_batch.cuda() 

        optim.zero_grad()
        outputs = LSTM(x_batch)

        loss = criterion(y_batch.flatten().cuda(), outputs.cuda() )
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        if iter % 50 == 0:

            x_batch, y_batch = dh.next_stock_batch(batch_size, n_steps, df_test)
            x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
            x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()
            with torch.no_grad():
                outputs = LSTM(x_batch)
                test_loss = criterion(y_batch.flatten().cuda(), outputs.cuda() )

            if best_val > test_loss:
                best_val = test_loss
                torch.save(LSTM, "Best_val_model.pth")
            print(iter, "\t Train Loss:", loss.item(), "\t Test Loss:", test_loss.item() )

    #plt.plot(train_loss, label= "Train Loss")
    #plt.xlabel(" Iteration ")
    #plt.ylabel("Loss value")
    #plt.legend(loc="upper left")
    #plt.show()
    #plt.clf()

train(501, df_train, df_test)
LSTM = torch.load("Best_val_model.pth")


x_batch, y_batch = dh.next_stock_batch(batch_size, n_steps, df_train)
x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()


if torch.cuda.is_available():
        x_batch.cuda()
        y_batch.cuda() 


with torch.no_grad():
    LSTM.eval()
    outputs = LSTM(x_batch)
    loss = criterion(y_batch.flatten().cuda(), outputs.cuda() )
    print(loss)

y = y_batch.cpu().numpy().reshape((batch_size,hidden_dim))[0,:]
o = outputs.cpu().numpy().reshape((batch_size,hidden_dim))[0,:]

#plt.plot(y, label= "Ground truth")
#plt.plot(o, label = "Prediction")
#plt.xlabel(" Time ")
#plt.ylabel("Stock return")
#plt.legend(loc="upper left")
#plt.savefig('seq1.png')
#plt.show()
#plt.clf()

x_batch, y_batch = dh.next_stock_batch(batch_size, n_steps, df_test)
x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()


if torch.cuda.is_available():
        x_batch.cuda()
        y_batch.cuda() 


with torch.no_grad():
    LSTM.eval()
    outputs = LSTM(x_batch)
    loss = criterion(y_batch.flatten().cuda(), outputs.cuda() )
    print(loss)

y = y_batch.cpu().numpy().reshape((batch_size,hidden_dim))[0,:]
o = outputs.cpu().numpy().reshape((batch_size,hidden_dim))[0,:]

#print(o)
#print(y)
print(outputs.shape)
plt.plot(y, label= "Ground truth")
plt.plot(o, label = "Prediction")
plt.xlabel(" Time ")
plt.ylabel("Stock return")
plt.legend(loc="upper left")
plt.savefig('seq2.png')
#plt.show()
plt.clf()