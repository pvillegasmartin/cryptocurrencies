import train
import torch
import data_handler as dh
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

LSTM = torch.load(f"Model{train.period}_ldim{train.layer_dim}_nsteps{train.n_steps}.pth")
df_train, df_test, scaler, data_train, data_test = dh.create_data()
max_range = df_test.shape[0]-train.n_steps*train.period_mins-1
predictions = []
LSTM.eval()

for i in range(max_range):
    print(f'{i+1} out of {max_range}')
    x_batch, y_batch = dh.next_stock_batch(1, df_test, train.period_mins, n_steps=7, starting_points=np.array([i]))
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
    x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()
    outputs = LSTM(x_batch)
    o = outputs.detach().numpy().reshape((1, train.output_dim)).flatten()
    o_copies = np.repeat(o, train.input_dim, axis=-1).reshape(1, -1)
    o = scaler.inverse_transform(o_copies)[:, 0]
    predictions.append(o.item())


df_plot = scaler.inverse_transform(df_test.iloc[train.period_mins*train.n_steps:,:])[:, 0]
plt.plot(df_plot, label="Ground truth")
plt.plot(predictions, label="Prediction")
plt.xlabel(" Time ")
plt.ylabel("RV return")
plt.legend(loc="upper left")
#plt.show()
plt.savefig(f'Test_2022_{train.period}_ldim{train.layer_dim}_nsteps{train.n_steps}.png')
plt.clf()