import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(input_size, 256)
        self.linear4 = nn.Linear(256, output_size)
        # self.dropout = nn.Dropout(0.3)
        # self.act_relu = nn.ReLU()

    def forward(self, x):
        x_cp = x
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x_cp = F.leaky_relu(self.linear3(x_cp))

        if x.shape != x_cp.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {x_cp.shape}")
        # x = self.dropout(F.relu(self.linear3(x)))
        x = x + x_cp

        x = self.linear4(x)
        return x

    def save(self, model_folder_path, file_name='model.pt'):

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.track_loss = []

    def train_step(self, state, action, reward, next_state, done, store_loss=False):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        
        ####################################################
        # Convert done to a tensor of 0/1 
        if isinstance(done, (list, tuple)): 
            done = torch.tensor(done, dtype=torch.float) 
        else: 
            done = torch.tensor([done], dtype=torch.float)
        ####################################################

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # done = (done, )
            done = torch.unsqueeze(done, 0)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)

        # store loss in case of long memory train
        if store_loss:
            self.track_loss.append(loss.item())
        loss.backward()

        self.optimizer.step()


