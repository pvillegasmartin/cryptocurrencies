import train
import torch
import data_handler as dh
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

df_train, df_test, scaler, data_train, data_test = dh.create_data(output=train.output_shift)
try:
    LSTM = torch.load(f"Model{train.period}_out{train.output_shift}_inputsdim{train.input_dim}_ldim{train.layer_dim}_nsteps{train.n_steps}.pth")
except:
    train.train(1000, df_train, df_test)
    LSTM = torch.load(f"Model{train.period}_out{train.output_shift}_inputsdim{train.input_dim}_ldim{train.layer_dim}_nsteps{train.n_steps}.pth")
max_range = df_test.shape[0] - train.n_steps - 1
predictions = []
LSTM.eval()

for i in range(max_range):
    print(f'{i+1} out of {max_range}')
    x_batch, y_batch = dh.next_stock_batch(1, df_test, n_steps=train.n_steps, starting_points=np.array([i]))
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
    x_batch, y_batch = Variable(x_batch).float(), Variable(y_batch).float()
    outputs = LSTM(x_batch)
    o = outputs.detach().numpy().reshape((1, train.output_dim)).flatten()
    o_copies = np.repeat(o, train.input_dim+1, axis=-1).reshape(1, -1)
    o = scaler.inverse_transform(o_copies)[:, 0]
    predictions.append(o.item())


df_pred = scaler.inverse_transform(df_test.iloc[train.n_steps+1:, :])[:, 0]
df_ground = scaler.inverse_transform(df_test.iloc[train.n_steps+1:, :])[:, 1]
plt.plot(df_ground, label="Ground truth")
#plt.plot(df_pred, label="To predict")
plt.plot(predictions, label="Prediction")
plt.xlabel(" Time ")
plt.ylabel("Output")
plt.legend(loc="upper left")
plt.show()
#plt.savefig(f'Test_2022_{train.period}_out{train.output_shift}_inputsdim{train.input_dim}_ldim{train.layer_dim}_nsteps{train.n_steps}.png')
#plt.clf()