### 1D CNN Training

###################################################################
# Code Author: Rama Sesha Sridhar Mantripragada \
# Email: rmantrip@gmu.edu

# This is main code of the CNN model. This code trains the CNN model and saves the model weights to h5 and text files \
# The CNN bass filtered data is saved to a .nc file
###
### Load all the required packages

import numpy as np
import xarray as xr
import copy
import multiprocessing
import tensorflow as tf
import os
import logging

### Define custom functions

def smooth_clim(data):   
    
    """
    Smooths the input climate data by calculating daily climatology and applying a low-pass
    Fourier filter to retain only the first few harmonic components of the annual cycle.

    Args:
    data (xarray.DataArray or xarray.Dataset): Input climate data with a "time" dimension.

    Returns:
        smclim (xarray.DataArray or xarray.Dataset): Smoothed climatology data with the same 
            dimensions and coordinates as the input data.

    Notes:
        - This function assumes that the input data is daily data and is evenly spaced in time.
        - The number of harmonics to retain is determined by the variable `nharm`. In this
          implementation, `nharm` is set to 3.
    """
    
    # Calculate daily climatology
    clim = data.groupby("time.dayofyear").mean("time")
    # smoothed annual cycle
    nharm = 3
    fft = np.fft.rfft(clim, axis=0)
    fft[nharm] = 0.5*fft[nharm]
    fft[nharm+1:] = 0
    dataout = np.fft.irfft(fft, axis=0)
    smclim = copy.deepcopy(clim)
    smclim[:] = dataout 

    return smclim


def filtwghts_lanczos(nwt, filt_type, fca, fcb):
	
    """
    Calculates the Lanczos filter weights.
    
    Parameters
    ----------
    nwt : int
        The number of weights.
    filt_type : str
        The type of filter. Must be one of 'low', 'high', or 'band'.
    fca : float
        The cutoff frequency for the low or band filter.
    fcb : float
        The cutoff frequency for the high or band filter.
    
    Returns
    -------
    w : ndarray
        The Lanczos filter weights.
    
    Notes
    -----
    The Lanczos filter is a type of sinc filter that is truncated at a specified frequency.
    This function implements a Lanczos filter in the time domain.
    """	
	
    pi = np.pi
    k = np.arange(-nwt, nwt+1)

    if filt_type == 'low':
        w = np.zeros(nwt*2+1)
        w[:nwt] = ((np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt))
        w[nwt+1:] = ((np.sin(2 * pi * fca * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt))
        w[nwt] = 2 * fca
    elif filt_type == 'high':
        w = np.zeros(nwt*2+1)
        w[:nwt] = -1 * (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w[nwt+1:] = -1 * (np.sin(2 * pi * fcb * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w[nwt] = 1 - 2 * fcb
    else:
        w1 = np.zeros(nwt*2+1)
        w1[:nwt] = (np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w1[nwt+1:] = (np.sin(2 * pi * fca * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w1[nwt] = 2 * fca
        w2 = np.zeros(nwt*2+1)
        w2[:nwt] = (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w2[nwt+1:] = (np.sin(2 * pi * fcb * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w2[nwt] = 2 * fcb
        w = w2 - w1
		
    return w


def remove_partial_year(da):
	
    """
    This function removes partial years from the beginning and end of a given xarray DataArray.
    
    Parameters:
    -----------
    da : xarray.DataArray
        The input data array that contains time-series data.
        
    Returns:
    --------
    xarray.DataArray
        The updated data array with partial years removed from the beginning and end.
    """
	
    start_year = da.time[0].dt.year.values
    end_year = da.time[-1].dt.year.values

    da = da.sel(time=slice(f"{start_year+1}-01-01", None))
    da = da.sel(time=slice(None, f"{end_year-1}-12-31"))
        
    return da


def cnn_bpf(cc, xtrain_split, ytrain_split, xval_split, yval_split, xtest_split, no_epochs, verbosity, kernel1, kernel2, ngp, outPath):
    
    """
    Trains a convolutional neural network model on the input training data and returns a bandpass filtered test data.

    Args:
        cc (int): Index of the parallel processing.
        xtrain_split (numpy array): Input training data for the current processor.
        ytrain_split (numpy array): Target training data for the current processor.
        xval_split (numpy array): Input validation data for the current processor.
        yval_split (numpy array): Target validation data for the current processor.
        xtest_split (numpy array): Input test data for the current processor.
        no_epochs (int): Number of epochs to train the model.
        verbosity (int): Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        kernel1 (int): Size of the kernel for the first depthwise convolution layer.
        kernel2 (int): Size of the kernel for the second depthwise convolution layer.
        ngp (int): Number of grid points in the input data.
        outPath (str): Output path to save the model weights.

    Returns:
        numpy array: Predictions on the test data for the current processor.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
    inputs = tf.keras.Input(shape=(None,ngp[cc]),batch_size=1,name='input_layer')
    smoth1 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel1,padding='same',use_bias=False,activation='linear')(inputs)
    diff = tf.keras.layers.subtract([inputs, smoth1])
    smoth2 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel2,padding='same',use_bias=False,activation='linear')(diff)
    model = tf.keras.Model(inputs=inputs, outputs=smoth2)
    model.compile(optimizer='adam', loss='mse')

    model.fit(xtrain_split[cc],ytrain_split[cc],epochs=no_epochs,validation_data=(xval_split[cc], yval_split[cc]),verbose=verbosity,callbacks=[callback])

    model.save_weights(outPath+str(cc)+'.h5')
    pred = model.predict(xtest_split[cc]).squeeze()
    
    return pred

def save_weights_text(kernel1, kernel2, ngp, var_name,outPath):
    
    """
    Loads saved CNN model weights and saves the filter weights to text files.

    Args:
        kernel1 (int): Size of the kernel for the first depthwise convolution layer.
        kernel2 (int): Size of the kernel for the second depthwise convolution layer.
        ngp (list): List of the number of grid points for each input data.
        var_name (str): Variable name to use for the output text files.

    Returns:
        None
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    filter1 = []
    filter2 = []

    for i, ng in enumerate(ngp):
        inputs = tf.keras.Input(shape=(None,ng),batch_size=1,name='input_layer')
        smoth1 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel1,padding='same',use_bias=False,activation='linear')(inputs)
        diff = tf.keras.layers.subtract([inputs, smoth1])
        smoth2 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel2,padding='same',use_bias=False,activation='linear')(diff)
        model = tf.keras.Model(inputs=inputs, outputs=smoth2)
        model.compile(optimizer='adam', loss='mse')
        model.load_weights(outPath + str(i) + '.h5')
        ex1 = model.layers[1].get_weights()[0].squeeze()
        ex2 = model.layers[3].get_weights()[0].squeeze()
        filter1.append(ex1)
        filter2.append(ex2)
        
    filter1 = np.hstack(filter1)
    filter2 = np.hstack(filter2)
    
    np.savetxt(outPath+var_name+'.filter1.txt', filter1, delimiter=',')
    np.savetxt(outPath+var_name+'.filter2.txt', filter2, delimiter=',') 



### Enter latitude and longitude bounds of the domain

lat_north, lat_south, long_west, long_east = 7.5, -7.5, 125, 270

### Load the NOAA interpolated data. 
#The NOAA interpolated data is downloaded from: https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc 

# NOAA interpolated daily mean OLR
olriFile = 'olr.day.i.mean.nc'
olri = xr.open_dataset(olriFile).sel(lat=slice(lat_north,lat_south),lon=slice(long_west,long_east),time=slice('1980',None)).olr

### Check if the input data contains any NaNs. Get the location indices of NaNs if present.

if olri.isnull().any().item():
    print("DataArray contains NaN values.")
    nan_locations = olri.isnull()
    nan_indices = np.argwhere(nan_locations.values)
else:
    print("DataArray does not contain NaN values.")

### Print the start year, end year, and total number of years in the data

start_year = olri.time[0].dt.year.values
end_year = olri.time[-1].dt.year.values
print('start_year: ', start_year)
print('end_year: ', end_year)
print('Total number of years: ',end_year - start_year + 1)

### Get the years to split the dataset into training, validation, and testing. 

startTrain, endTrain, startVal, endVal, startTest, endTest = '1988', '2012', '2013', '2014', '2015', '2016'

### Calculate climatology based only on the training period. This will avoid test data leakages or bias.
### Calculate unfiltered anomalies and band pass filtered anomalies 

# Calculate the climatology

olri_clim = olri.sel(time=slice(startTrain, endTrain)).groupby("time.dayofyear").mean("time")

# Calculate the anomalies

olri_anom = olri.groupby("time.dayofyear") - olri_clim

# Calculate 30-90-day band pass Lanczos filterd olr anomalies 

nwgths = 90 # The filter uses 181 (nwgths*2+1) weights
tpa = 30 # Time period: 30-day
tpb = 90 # Time period: 90-day
wgths = filtwghts_lanczos(nwgths,'band',1/tpb,1/tpa)
wgths = xr.DataArray(wgths, dims=['window'])

olri_bpf = olri_anom.rolling(time=len(wgths), center=True).construct('window').dot(wgths)

### Remove Partial Year data
#The band pass filtered data will have 90 days of missing data in the begining and end of the period. \
#The number of missing days is equal to nwgths. The CNN model input data should not contain any missing data.\
#Therefore, we will delete the first and last 90 days data for both olri_anom and olri_bpf and also remove partial year data.

# Get the time dimension
time = olri_bpf.time

# Calculate the number of days in the time dimension
time_length = (time[-1]-time[0]).dt.days + 1

# Select only the data from 90 days after the start of the time dimension to 90 days before the end
olri_bpf = olri_bpf.sel(time=slice(time[90], time[time_length-91]))
olri_anom = olri_anom.sel(time=slice(time[90], time[time_length-91]))

# Remove the partial year data
olri_bpf = remove_partial_year(olri_bpf)
olri_anom = remove_partial_year(olri_anom)

### Combine the latitude and longitude cordinates to a single "grid" dimension 

# Combine and transpose the data to (time, grid)

olri_anom_rg = olri_anom.stack(grid=("lat","lon")).transpose("time","grid")
olri_bpf_rg = olri_bpf.stack(grid=("lat","lon")).transpose("time","grid")

### Split the dataset to training, validation, and testing

xtrain = olri_anom_rg.sel(time=slice(startTrain, endTrain)).expand_dims(dim='new',axis=0).values
ytrain = olri_bpf_rg.sel(time=slice(startTrain, endTrain)).expand_dims(dim='new',axis=0).values

xval = olri_anom_rg.sel(time=slice(startVal, endVal)).expand_dims(dim='new',axis=0).values
yval = olri_bpf_rg.sel(time=slice(startVal, endVal)).expand_dims(dim='new',axis=0).values

xtest = olri_anom_rg.sel(time=slice(startTest, endTest)).expand_dims(dim='new',axis=0).values
ytest = olri_bpf_rg.sel(time=slice(startTest, endTest))

### Enter below the number of cores (cpus) to be alloted for parallel processing

num_cpus = 12

### Split the data along grid dimension to pass to each core for parallel processing 
#This will speed up the computations

xtrain_split = np.array_split(xtrain,num_cpus, axis=2)
ytrain_split = np.array_split(ytrain,num_cpus, axis=2)
xval_split = np.array_split(xval,num_cpus, axis=2)
yval_split = np.array_split(yval,num_cpus, axis=2)
xtest_split = np.array_split(xtest,num_cpus, axis=2)

ngp = np.ones(num_cpus, dtype='int')
ngp = [xtrain_split[i].shape[2] for i in range(num_cpus)]

### Train the CNN model, get the CNN band pass filtered data, and save the weights to a text file

# Set the output path to store the model weights
outPath = '/homes/rmantrip/testPyISV/cnnweights/olr/'

kernel1 = 90
kernel2 = 30
no_epochs = 500
verbosity = 0
var_name = 'olr'

def wrapper_cnn_bpf(args):
    return cnn_bpf(*args)

if __name__ == '__main__':
    # Define the arguments for each process
    arguments = [(cc, xtrain_split, ytrain_split, xval_split, yval_split, xtest_split, no_epochs, verbosity, kernel1, kernel2, ngp, outPath) for cc in range(num_cpus)]

    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = np.hstack(pool.map(wrapper_cnn_bpf, arguments))

# Save weight to a text file
save_weights_text(kernel1,kernel2,ngp,var_name,outPath)

# # Save the CNN filtered data to a netcdf file
cnn_bpf1 = copy.deepcopy(ytest)
cnn_bpf1[:] = np.nan
cnn_bpf1[:] = results
cnn_bpf1 = cnn_bpf1.unstack()
cnn_bpf1 = cnn_bpf1.rename(var_name)
cnn_bpf1.to_netcdf(outPath+'olr.cnn.bpf.nc')