import numpy as np

Nr = 30
Ne = 15000
batch_size = 8
neg_per_positive = 10
h_batch, r_batch, t_batch = np.random.randint(0, Ne, size=(batch_size,)), np.random.randint(0, Nr, size=(batch_size,)),np.random.randint(0, Ne, size=(batch_size,))
print('h_batch', h_batch, 'r_batch', r_batch, 't_batch', t_batch)

neg_hn_batch = np.random.randint(0, Ne, size=(batch_size, neg_per_positive))  # random index without filtering
print('neg_hn_batch', neg_hn_batch.shape)
neg_rel_hn_batch = np.tile(r_batch, (neg_per_positive, 1)).transpose()  # copy
print('neg_rel_hn_batch', neg_rel_hn_batch.shape)
neg_t_batch = np.tile(t_batch, (neg_per_positive, 1)).transpose()
print('neg_rel_hn_batch', neg_rel_hn_batch.shape)