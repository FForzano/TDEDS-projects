import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tikzplotlib as tpl

from utils import QMSE, CMSE

show_plot = False

try:
    if 'show' in sys.argv[1]:
        show_plot = True
except Exception:
    pass

colors = ['C0', 'C1', 'C2']

dataset = np.load('QDataset_v1.npy')

# linearize dataset
dataset = dataset.reshape(np.shape(dataset)[0]*np.shape(dataset)[1],np.shape(dataset)[2])

# normalize dataset
dataset_mean = np.full(np.shape(dataset.T), np.mean(dataset, axis=1)).T
dataset -= dataset_mean
# dataset_var = np.full(np.shape(dataset.T), np.var(dataset, axis=1)).T
# # dataset_var[np.abs(dataset_var)==0] = 10e-5
# dataset /= dataset_var
# dataset[np.isnan(dataset)] = 0

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


outFile = open('Eigenvalues_dimension_criterium_1.txt', 'wt')
eig_progression = np.cumsum(eigenvalues)

if show_plot:
    plt.plot(range(1,len(eigenvalues)+1), eig_progression/eig_progression[-1]*100)
    plt.xlabel('number of features')
    plt.ylabel('percentage of variance covered')
    tpl.save('Eigenvalues_dimension_criterium_1.tex')
    plt.show()

for i in range(len(eig_progression)):
    print(i+1, eig_progression[i]/eig_progression[-1]*100, file=outFile)
outFile.close()


# MSE criterion
MSE = []
Class_MSE = []
outFile = open('MSE_dimension_criterium_1.txt', 'wt')
class_outFile = open('CMSE_dimension_criterium_1.txt', 'wt')

for i in range(len(eigenvalues)):
    main_components = eigenvectors[:,range(i+1)]
    reduced_dataset = dataset.conj().T @ main_components
    reduced_dataset = np.real(reduced_dataset)

    reconstructed_datas = main_components @ reduced_dataset.T
    # de-normalization
    # reconstructed_datas *= dataset_var
    reconstructed_datas += dataset_mean

    # reshaperd_dataset = np.reshape((dataset*dataset_var)+dataset_mean,(int(np.math.sqrt(np.shape(dataset)[0])), int(np.math.sqrt(np.shape(dataset)[0])),np.shape(dataset)[1] ))
    reshaperd_dataset = np.reshape(dataset+dataset_mean,(int(np.math.sqrt(np.shape(dataset)[0])), int(np.math.sqrt(np.shape(dataset)[0])),np.shape(dataset)[1] ))
    reshaped_recDatas = np.reshape(reconstructed_datas,(int(np.math.sqrt(np.shape(reconstructed_datas)[0])), int(np.math.sqrt(np.shape(reconstructed_datas)[0])),np.shape(reconstructed_datas)[1] ))
    MSE.append(QMSE(reshaperd_dataset, reshaped_recDatas) )
    Class_MSE.append(CMSE(reshaperd_dataset, reshaped_recDatas))

    if i>=20:
        break

for i in range(len(MSE)):
    print(i+1, MSE[i], file=outFile)
    print(i+1, Class_MSE[i], file=class_outFile)
outFile.close()
class_outFile.close()
# tpl.save('MSE_dimension_criterium_1.tex')

if show_plot:
    plt.plot(MSE)
    plt.show()
    plt.plot(Class_MSE)
    plt.show()

### PCA ALGORITHM ###

## PCA with d=2
print('PCA with 2 principal components')

main_components = eigenvectors[:,[0,1]]
reduced_dataset = dataset.conj().T @ main_components
reduced_dataset = np.real(reduced_dataset)


if show_plot:
    fig2D, ax2D = plt.subplots()

dataOut = open('PlotDatas/2levelCoherent_2PCs.txt', 'wt')

i=0
for reducted_data in reduced_dataset:
    if show_plot:
        ax2D.scatter(reducted_data[0], reducted_data[1], color=colors[int(i/20)])
    print(reducted_data[0], reducted_data[1], file=dataOut)
    i+=1

dataOut.close()

if show_plot:
    fig2D.legend()
    fig2D.show()


# ## PCA with d=3
print('PCA with 3 principal components')

if show_plot:
    fig3D = plt.figure()
    ax3D = mplot3d.Axes3D(fig3D)

main_components = eigenvectors[:,[0,1,2]]
reduced_dataset = dataset.conj().T @ main_components
reduced_dataset = np.real(reduced_dataset)

dataOut = open('PlotDatas/2levelCoherent_3PCs.txt', 'wt')

i=0
for reducted_data in reduced_dataset:
    if show_plot:
        if i == 0:
            ax3D.scatter3D(reducted_data[0], reducted_data[1], reducted_data[2], color=colors[int(i/20)], label='coherent')
        elif i == 10:
            ax3D.scatter3D(reducted_data[0], reducted_data[1], reducted_data[2], color=colors[int(i/20)], label='thermal')
        else:
            ax3D.scatter3D(reducted_data[0], reducted_data[1], reducted_data[2], color=colors[int(i/20)])
    print(reducted_data[0], reducted_data[1], reducted_data[2], file=dataOut)
    i+=1

dataOut.close()
if show_plot:
    ax3D.legend()
    fig3D.show()

input('Press a key for terminate the simulation...')