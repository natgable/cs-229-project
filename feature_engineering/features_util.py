import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pywt
from sklearn.decomposition import PCA

def dft_features(X):
    """
        Generate features using the 2-D Discrete Fourier Transform on the input images.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples, dim).
    """
    N_train = X.shape[0]
    nchannels = 3
    X_dft = np.zeros(X.shape)
    print("generating dft features")
    for image in tqdm.tqdm(range(N_train)):
        for channel in range(nchannels):
            X_dft[image, :, :, channel] = np.real(np.fft.fft2(X[image, :, :, channel]))
            
    return X_dft


def wavelet_features(X):
    """
        Generate features using the 2-D Discrete Wavelet Transform (similar to DCT) on the input images.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples, dim).
    """
    N_train = X.shape[0]
    nchannels = 3
    X_dwt = np.zeros(X.shape)
    print("generating wavelet features")
    for image in tqdm.tqdm(range(N_train)):
        for channel in range(nchannels):
            coeffs = pywt.dwt2(X[image, :, :, channel], 'haar')
            LL, (LH, HL, HH) = coeffs
            X_dwt[image, :, :, channel] = np.block([[LL, LH], [HL, HH]])
    
    return X_dwt


def pca_features(X, n_components=0.9):
    """
        Peform PCA on the input features. Does PCA channel-by-channel for RBG images.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples, dim).
    """
    nchannels = 3
    N_train = X.shape[0]
    img_size = X.shape[1]
    
    pca_out = np.zeros(X.shape)
    
    for channel in range(nchannels):
        X_channel = X[:,:,:,channel].reshape(N_train, img_size * img_size)

        train_pca_channel = PCA(n_components=n_components)
        train_pca_channel.fit(X_channel)
        components = train_pca_channel.transform(X_channel)
        projected_channel = train_pca_channel.inverse_transform(components)
        
        pca_out[:,:,:,channel] = projected_channel.reshape(N_train, img_size, img_size)
    
    return pca_out
    

