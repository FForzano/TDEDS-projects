import dataset_gen as dataset
from utils import *

data = None

dataset.set_noise_parameter(10e-2, 10e-1)

N = 50

n_sample = 20

new_datas=None

mu = 1.5
new_datas = dataset.generate_dataset(num=n_sample, N=N, state='coherent', mean=mu)
data = new_datas

mu = 2.5
new_datas = dataset.generate_dataset(num=n_sample, N=N, state='coherent', mean=mu)
temp_data = data
data = np.empty((N,N, np.shape(temp_data)[2]+np.shape(new_datas)[2]), dtype='complex_')
data[:,:, :np.shape(temp_data)[2]] = temp_data
data[:,:, -np.shape(new_datas)[2]:] = new_datas

new_datas = dataset.generate_dataset(num=n_sample, N=N, state='thermal')
temp_data = data
data = np.empty((N,N, np.shape(temp_data)[2]+np.shape(new_datas)[2]), dtype='complex_')
data[:,:, :np.shape(temp_data)[2]] = temp_data
data[:,:, -np.shape(new_datas)[2]:] = new_datas

# data = shuffle_along_axis(data,axis=2)

np.save('QDataset_v2', data)