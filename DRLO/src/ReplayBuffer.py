from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        # self.buffer = []
        # self.min_reward = 999999
        # self.min_index = -1
        # self.index = -1
    # 获取batsize的历史经验
    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return self.sample(self.num_experiences)
        else:
            return self.sample(batch_size)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        actions = np.array(actions).reshape(batch_size, -1)
        rewards = np.array(rewards).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done


    def is_full(self):
        if self.buffer_size == self.num_experiences:
            return True
        else:
            return False

    def getSize(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        random.shuffle(self.buffer)
        experience = (state, action, reward, new_state, done)
        # experience = [state, action, reward, new_state, done]
        # print(experience)
        if self.num_experiences < self.buffer_size:
            # self.index += 1
            # if reward < self.min_reward:
            #     self.min_reward = reward
            #     self.min_index = self.index
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            # self.buffer = deque(sorted(self.buffer, key=lambda temp: temp[2]))
            self.buffer = deque(sorted(self.buffer, key=lambda temp: temp[2]))
            # self.buffer.popleft()
            # self.buffer.append(experience)
            experience_min = self.buffer.popleft()

            if experience_min[2] < experience[2]:
            # if experience_max[2] > experience[2]:
                self.buffer.append(experience)
            else:
                self.buffer.append(experience_min)


            # experience_min = (deque(sorted(self.buffer, key=lambda temp: temp[2]))).popleft()
            # print(experience_min)
            # reward_min = experience_min[2]
            # if reward_min < experience[2]:
            #     self.buffer.remove(experience_min)
                # self.buffer.index(experience_min)
                # self.buffer.append(experience)

    def getCount(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def saveBuffer(self):
        file_path = "src\\buffer"
        with open(file_path, "w") as f:
            for item in self.buffer:
                s = str(item[0]) + str(item[1]) + str(item[2]) + str(item[3]) + str(item[4])
                f.write(s)