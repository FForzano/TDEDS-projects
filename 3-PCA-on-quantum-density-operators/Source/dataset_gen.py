import qutip
import numpy as np

noise_mean = 10**(-2)
noise_deviation = noise_mean/10

def set_noise_parameter(mean, deviation):
    global noise_mean
    global noise_deviation

    noise_mean = mean
    noise_deviation = deviation


def generate_dataset(num=10, N=30, state='coherent', mean=0.3, squeeze=0.2):
    if state == 'coherent':
        th_noise = np.abs(np.random.normal(loc=noise_mean, scale=noise_deviation, size=num))

        states = np.empty((N,N,num), dtype='complex_')
        for i in range(num):
            states[:,:,i] = qutip.displace(N,mean) * qutip.thermal_dm(N,th_noise[i]) * qutip.displace(N,mean).conj()
            states[:,:,i] /= np.trace(states[:,:,i])

    if state == 'thermal':
        th_noise = np.abs(np.random.normal(loc=noise_mean, scale=noise_deviation, size=num))

        states = np.empty((N,N,num), dtype='complex_')
        for i in range(num):
            states[:,:,i] = qutip.thermal_dm(N,th_noise[i])

    if state == 'squeezed':
        squeeze = squeeze * np.exp(1j*np.pi)
        th_noise = np.abs(np.random.normal(loc=noise_mean, scale=noise_deviation, size=num))

        states = np.empty((N,N,num), dtype='complex_')
        for i in range(num):
            states[:,:,i] = qutip.squeeze(N,squeeze) * qutip.displace(N,mean) * qutip.thermal_dm(N,th_noise[i]) * qutip.displace(N,mean).conj() * qutip.squeeze(N,squeeze).conj()
            states[:,:,i] /= np.trace(states[:,:,i])

    return states

if __name__ == '__main__':
    dataset = None

    set_noise_parameter(10e-2, 10e-1)

    N = int(input('Insert the Hilbert space dimension:\t'))

    try:
        while True:
            state_type = input('Insert the quantum state type (coherent, squeezed, thermal):\t')
            state_type = state_type.strip()

            n_sample = int( input('Insert the number of samples for this type of states:\t') )

            new_datas=None

            if state_type == 'coherent':
                mu = float( input('insert the displacement paramenter:\t') )
                new_datas = generate_dataset(num=n_sample, N=N, state=state_type, mean=mu)
            else:
                new_datas = generate_dataset(num=n_sample, N=N, state=state_type)

            if dataset is None:
                dataset = new_datas
            else:
                temp_dataset = dataset
                dataset = np.empty((N,N, np.shape(temp_dataset)[2]+np.shape(new_datas)[2]), dtype='complex_')
                dataset[:,:, :np.shape(temp_dataset)[2]] = temp_dataset
                dataset[:,:, -np.shape(new_datas)[2]:] = new_datas

    except KeyboardInterrupt:
        print('Writing the dataset')

    np.save('QDataset', dataset)

