import copy
from collections import deque
import numpy as np
import torch
from environment.environment import Environment, Action, initialize_environment
from IPython.display import clear_output
import random
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

now = datetime.now()
time = now.strftime("%H:%M:%S")
writer = SummaryWriter(f"runs/test_name/{time}")

grid_width, grid_height = (4, 4)
l1 = 72
l2 = 512
l3 = 128
l4 = 128
l5 = 64
l6 = 6

action_set = {
    0: Action(1),
    1: Action(2),
    2: Action(3),
    3: Action(4),
    4: Action(5),
    5: Action(6),
}

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.ReLU(),
    torch.nn.Linear(l4, l5),
    torch.nn.ReLU(),
    torch.nn.Linear(l5, l6),
)

model2 = copy.deepcopy(model)  # A
model2.load_state_dict(model.state_dict())  # B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

gamma = 0.7
# epsilon = 0.3

epochs = 10000
losses = []
mem_size = 1000
batch_size = 32
replay = deque(maxlen=mem_size)
max_moves = 50
h = 0
sync_freq = 500  # A
j = 0

# writer.add_graph(model)
for i in range(epochs):
    pre_states_2 = []

    if i < epochs // 3:
        pit_proba = 0.0
        epsilon = 0.5
    elif i < epochs // 1.5:
        pit_proba = 0.05
        epsilon = 0.3
    elif i < epochs // 1.75:
        pit_proba = 0.1
        epsilon = 0.2
    else:
        pit_proba = 0.2
        epsilon = 0.1

    next_environment, initial_percept = initialize_environment(
        grid_width=grid_width,
        grid_height=grid_height,
        pit_proba=pit_proba,
        allow_climb_without_gold=False,
    )
    print("pit ", pit_proba)
    state1_ = next_environment.agent.q_transform()  # + np.random.rand(1, 72)/100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0

    while status == 1:
        j += 1
        mov += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        r_int = random.random()
        if r_int < epsilon:
            action_ = np.random.randint(0, l6)  # l6
            if round(r_int * 100) % 2 == 0:
                action_ = 0
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        # print("recommended action", action)

        next_environment.agent = next_environment.agent.apply_move_action(action, 4, 4)
        next_environment, next_percept = next_environment.apply_action(action)
        state2_ = next_environment.agent.q_transform()  # + np.random.rand(1, 72)/100.0
        state2 = torch.from_numpy(state2_).float()
        if any(np.array_equal(x, state2_) for x in pre_states_2):
            reward = -1000
        else:
            reward = next_percept.reward
            pre_states_2.append(state2_)

        if reward > 0:
            print("positive reward", reward, action)

        done = True if (reward > 0) or (reward == -1000) else False
        exp = (state1, action_, reward, state2, done)
        replay.append(exp)  # H
        state1 = state2
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = model2(state2_batch)  # B

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, round(loss.item(), 2))
            print("action: ", action)
            writer.add_scalar("train/loss", loss.item(), i)
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            torch.save(model, "./models")

            if j % sync_freq == 0:  # C
                model2.load_state_dict(model.state_dict())
        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0

losses = np.array(losses)

writer.close()
