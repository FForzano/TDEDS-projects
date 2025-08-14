
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

colors = ['C0', 'C1', 'C2']

dataset = np.load('QDataset_v2.npy')

# linearize dataset
dataset = dataset.reshape(np.shape(dataset)[0]*np.shape(dataset)[1],np.shape(dataset)[2])

# normalize dataset
# dataset -= np.mean(dataset, axis=0)
# dataset /= np.var(dataset, axis=0)

### COVARIANCE MATRIX & EIGENVALUE EVALUATION ###

print('Evaluation of the eigenvalues')
covariance_matrix = dataset @ dataset.conj().T
# eigenvalues = np.linalg.svd(covariance_matrix, compute_uv=False, hermitian=True)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# the covariance matrix is hermitian. Its eigenvalues are real
eigenvalues = np.real(eigenvalues)

eigenvalues = -eigenvalues
idx = eigenvalues.argsort()  
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
eigenvalues = -eigenvalues

eig_progression = np.cumsum(eigenvalues)
plt.plot(range(1,len(eigenvalues)+1), eig_progression/eig_progression[-1]*100)
plt.xlabel('number of features')
plt.ylabel('percentage of variance covered')
plt.show()


### PCA ALGORITHM ###

## PCA with d=2
print('PCA with 2 principal components')

main_components = eigenvectors[:,[0,1]]
reduced_dataset = dataset.conj().T @ main_components
reduced_dataset = np.real(reduced_dataset)

# plt.title('Eigenvectors')
# plt.plot(freq_axis, pca.components_[0], alpha = 0.5)
# plt.plot(freq_axis, pca.components_[1], alpha = 0.5)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Amplitude')
# plt.show()

fig2D, ax2D = plt.subplots()

# dataOut = open('PlotDatas/3levelCoherent_2PCs.txt', 'wt')

i=0
for reducted_data in reduced_dataset:
    ax2D.scatter(reducted_data[0], reducted_data[1], color=colors[int(i/20)])
    # print(reducted_data[0], reducted_data[1], file=dataOut)
    i+=1

# dataOut.close()

fig2D.show()


# ## PCA with d=3
print('PCA with 3 principal components')

fig3D = plt.figure()
ax3D = mplot3d.Axes3D(fig3D)

main_components = eigenvectors[:,[0,1,2]]
reduced_dataset = dataset.conj().T @ main_components
reduced_dataset = np.real(reduced_dataset)

# dataOut = open('PlotDatas/3levelCoherent_3PCs.txt', 'wt')

i=0
for reducted_data in reduced_dataset:
    ax3D.scatter3D(reducted_data[0], reducted_data[1], reducted_data[2], color=colors[int(i/20)])
    # print(reducted_data[0], reducted_data[1], reducted_data[2], file=dataOut)
    i+=1

# dataOut.close()
fig3D.show()

input('Press a key for terminate the simulation...')