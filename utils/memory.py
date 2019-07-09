import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = int(maxlen)
        self.start = 0
        self.length = 0
        self.data = np.zeros((self.maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    '''
    obs0 = {'minimap': [], 'screen': [], 'nonspatial': []}
    actions = {'categorical': [], 'screen1': [], 'screen2': []}
    obs1 = {'minimap': [], 'screen': [], 'nonspatial': []}
    '''
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = int(limit)
        self.action_shape = action_shape
        self.observation_shape = observation_shape

        # observation 0
        self.observations0_minimap = RingBuffer(self.limit, shape=self.observation_shape['minimap'])
        self.observations0_screen = RingBuffer(self.limit, shape=self.observation_shape['screen'])
        self.observations0_nonspatial = RingBuffer(self.limit, shape=self.observation_shape['nonspatial'])
        # action
        self.actions_categorial = RingBuffer(self.limit, shape=self.action_shape['categorical'])
        self.actions_screen1 = RingBuffer(self.limit, shape=self.action_shape['screen1'])
        self.actions_screen2 = RingBuffer(self.limit, shape=self.action_shape['screen2'])
        # reward & terminal flag
        self.rewards = RingBuffer(self.limit, shape=(1,))
        self.terminals1 = RingBuffer(self.limit, shape=(1,))
        # observation 1
        self.observations1_minimap = RingBuffer(self.limit, shape=self.observation_shape['minimap'])
        self.observations1_screen = RingBuffer(self.limit, shape=self.observation_shape['screen'])
        self.observations1_nonspatial = RingBuffer(self.limit, shape=self.observation_shape['nonspatial'])
        super().__init__()

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = {'minimap': self.observations0_minimap.get_batch(batch_idxs),
                      'screen': self.observations0_screen.get_batch(batch_idxs),
                      'nonspatial': self.observations0_nonspatial.get_batch(batch_idxs)}
        obs1_batch = {'minimap': self.observations1_minimap.get_batch(batch_idxs),
                      'screen': self.observations1_screen.get_batch(batch_idxs),
                      'nonspatial': self.observations1_nonspatial.get_batch(batch_idxs)}
        action_batch = {'categorical': self.actions_categorial.get_batch(batch_idxs),
                        'screen1': self.actions_screen1.get_batch(batch_idxs),
                        'screen2': self.actions_screen2.get_batch(batch_idxs)}
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': {'minimap': array_min2d(obs0_batch['minimap']),
                     'screen': array_min2d(obs0_batch['screen']),
                     'nonspatial': array_min2d(obs0_batch['nonspatial'])},
            'obs1': {'minimap': array_min2d(obs1_batch['minimap']),
                     'screen': array_min2d(obs1_batch['screen']),
                     'nonspatial': array_min2d(obs1_batch['nonspatial'])},
            'rewards': array_min2d(reward_batch),
            'actions': {'categorical': array_min2d(action_batch['categorical']),
                        'screen1': array_min2d(action_batch['screen1']),
                        'screen2': array_min2d(action_batch['screen2'])},
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        # obs0
        self.observations0_minimap.append(obs0['minimap'])
        self.observations0_screen.append(obs0['screen'])
        self.observations0_nonspatial.append(obs0['nonspatial'])
        # action
        self.actions_categorial.append(action['categorical'])
        self.actions_screen1.append(action['screen1'])
        self.actions_screen2.append(action['screen2'])
        # reward
        self.rewards.append(reward)
        # obs1
        self.observations1_minimap.append(obs1['minimap'])
        self.observations1_screen.append(obs1['screen'])
        self.observations1_nonspatial.append(obs1['nonspatial'])
        # terminal1
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return self.rewards.length


class EpisodeMemory(Memory):
    def __init__(self, limit, action_shape, observation_shape):
        super().__init__(limit, action_shape, observation_shape)

    def sample(self):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.array([x for x in range(self.nb_entries)])
        obs0_batch = {'minimap': self.observations0_minimap.get_batch(batch_idxs),
                      'screen': self.observations0_screen.get_batch(batch_idxs),
                      'nonspatial': self.observations0_nonspatial.get_batch(batch_idxs)}
        obs1_batch = {'minimap': self.observations1_minimap.get_batch(batch_idxs),
                      'screen': self.observations1_screen.get_batch(batch_idxs),
                      'nonspatial': self.observations1_nonspatial.get_batch(batch_idxs)}
        action_batch = {'categorical': self.actions_categorial.get_batch(batch_idxs),
                        'screen1': self.actions_screen1.get_batch(batch_idxs),
                        'screen2': self.actions_screen2.get_batch(batch_idxs)}
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': {'minimap': array_min2d(obs0_batch['minimap']),
                     'screen': array_min2d(obs0_batch['screen']),
                     'nonspatial': array_min2d(obs0_batch['nonspatial'])},
            'obs1': {'minimap': array_min2d(obs1_batch['minimap']),
                     'screen': array_min2d(obs1_batch['screen']),
                     'nonspatial': array_min2d(obs1_batch['nonspatial'])},
            'rewards': array_min2d(reward_batch),
            'actions': {'categorical': array_min2d(action_batch['categorical']),
                        'screen1': array_min2d(action_batch['screen1']),
                        'screen2': array_min2d(action_batch['screen2'])},
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def clear(self):
        # observation 0
        self.observations0_minimap = RingBuffer(self.limit, shape=self.observation_shape['minimap'])
        self.observations0_screen = RingBuffer(self.limit, shape=self.observation_shape['screen'])
        self.observations0_nonspatial = RingBuffer(self.limit, shape=self.observation_shape['nonspatial'])
        # action
        self.actions_categorial = RingBuffer(self.limit, shape=self.action_shape['categorical'])
        self.actions_screen1 = RingBuffer(self.limit, shape=self.action_shape['screen1'])
        self.actions_screen2 = RingBuffer(self.limit, shape=self.action_shape['screen2'])
        # reward & terminal flag
        self.rewards = RingBuffer(self.limit, shape=(1,))
        self.terminals1 = RingBuffer(self.limit, shape=(1,))
        # observation 1
        self.observations1_minimap = RingBuffer(self.limit, shape=self.observation_shape['minimap'])
        self.observations1_screen = RingBuffer(self.limit, shape=self.observation_shape['screen'])
        self.observations1_nonspatial = RingBuffer(self.limit, shape=self.observation_shape['nonspatial'])


if __name__ == '__main__':
    mem = Memory(limit=1e2,
                 action_shape={'categorical': (10,),
                               'screen1': (1, 32, 32),
                               'screen2': (1, 32, 32)},
                 observation_shape={'minimap': (7, 32, 32),
                                    'screen': (17, 32, 32),
                                    'nonspatial': (10,)})
    print(1+1)

    mem2 = EpisodeMemory(limit=1e2,
                         action_shape={'categorical': (10,),
                                       'screen1': (1, 32, 32),
                                       'screen2': (1, 32, 32)},
                         observation_shape={'minimap': (7, 32, 32),
                                            'screen': (17, 32, 32),
                                            'nonspatial': (10,)})

    print(1 + 1)
    mem2.clear()
    mem2.nb_entries
