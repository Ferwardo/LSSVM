import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class FE:
    def __init__(self, param):
        # Compute additional parameters
        param['stepSize'] = np.ceil((param['stepSize_ms']*param['samFreq'])/1000)
        param['frameSize'] = np.ceil((param['frameSize_ms']*param['samFreq'])/1000)
        param['fftSize'] = 2**np.ceil(np.log2(param['frameSize']))
        # Store the parameters
        self.param = param

    def dc_removal(self, data):
        if self.param['dcRemoval'] == 'hpf':
            # Do a HPF filtering with an IIR-filter
            data_f = signal.lfilter([1, -1], [1, -0.999], np.concatenate((np.zeros(1), data), axis=-1), axis=-1)
            # Remove the first sample of the data (due to filter delay)
            data_f = data_f[1:]
        elif self.param['dcRemoval'] == 'mean':
            # Remove mean from the data
            data_f = data - np.mean(data)
        return data_f

    def create_hamming_window(self):
        self.hamwin = np.hamming(self.param['frameSize']).astype(float)

    def create_sparse_mel_matrix_part1_part2(self):
        # Define the Mel frequency of high_freq and low_freq
        low_freq_mel = (2595 * np.log10(1 + self.param['lowFreq'] / 700))                           # LowMel
        high_freq_mel = (2595 * np.log10(1 + self.param['highFreq'] / 700))                         # NyqMel
        # Define the start Mel frequencies, start frequencies and start bins
        start_freq_mel = low_freq_mel + np.arange(0, self.param['melSize'], 1) / (self.param['melSize'] + 1) * (
                    high_freq_mel - low_freq_mel)                                                   # StartMel
        start_freq_hz = 700 * (10 ** (start_freq_mel / 2595) - 1)                                   # StartFreq
        # Define the stop Mel frequencies, start frequencies and start bins
        stop_freq_mel = low_freq_mel + np.arange(2, self.param['melSize'] + 2, 1) / (self.param['melSize'] + 1) * (
                    high_freq_mel - low_freq_mel)                                                   # StopMel
        stop_freq_hz = 700 * (10 ** (stop_freq_mel / 2595) - 1)                                     # StopFreq

        # get bins
        self.start_bin = np.round(self.param['fftSize'] / self.param['samFreq'] * start_freq_hz)     # melStartBin
        self.stop_bin = np.round(self.param['fftSize'] / self.param['samFreq'] * stop_freq_hz)       # melStopBin
        # The middle bins of the filters are the start frequencies of the next filter.
        self.middle_bin = np.append(self.start_bin[1:], self.stop_bin[-2])                           # melMiddle
        # Compute the width of the filters
        tot_len = self.stop_bin - self.start_bin + 1                                                 # TotLen (not required as externel parameter)
        self.low_len = self.middle_bin - self.start_bin + 1                                          # melLowLen
        self.high_len = tot_len - self.low_len + 1                                                   # melHiLen
        self.sum_to_len = int(np.sum(tot_len))

        # Compute the full filterbank
        self.full_mel_scale_vec = np.zeros((self.param['melSize'], int(np.floor(self.param['fftSize'] / 2 + 1))))
        # Compute the filter weights matrix
        for m in range(1, self.param['melSize'] + 1):
            weights_low = np.arange(1, self.low_len[m - 1] + 1) / (self.low_len[m - 1])
            for k in range(0, int(self.low_len[m - 1])):
                self.full_mel_scale_vec[m - 1, int(self.start_bin[m - 1] + k)] = weights_low[k]
            weights_high = np.arange(self.high_len[m - 1], 0, -1) / (self.high_len[m - 1])
            for k in range(0, int(self.high_len[m - 1])):
                self.full_mel_scale_vec[m - 1, int(self.middle_bin[m - 1] + k)] = weights_high[k]

        # Convert to a sparse filterbank
        self.sparse_mel_scale_vec = np.zeros(self.sum_to_len)
        k = 0
        for m in range(0, self.param['melSize']):
            for n in range(0, int(np.floor(self.param['fftSize'] / 2 + 1))):
                if self.full_mel_scale_vec[m, n]:
                    self.sparse_mel_scale_vec[k] = self.full_mel_scale_vec[m,n]
                    k = k + 1

    def framing(self, data):
        axis = 0
        dataSize = np.shape(data)[0]
        num_frames = int(np.floor(((dataSize - (self.param['frameSize'] - 1) - 1) / self.param['stepSize']) + 1))
        assert num_frames>0, 'num of frames is 0 in Framing()'
        indices = np.tile(np.arange(0, int(self.param['frameSize'])), (num_frames, 1)) + np.tile(np.arange(0, num_frames * int(self.param['stepSize']), int(self.param['stepSize'])), (int(self.param['frameSize']), 1)).T
        frames = np.take(data, indices, axis=axis)
        # Hamming operation
        frames *= np.reshape(self.hamwin, [(frames.shape[k] if k == (axis + 1) else 1) for k in range(len(frames.shape))])
        return frames

    def fft(self, frames):
        axis = 1
        fftframes = np.fft.rfft(frames, n=int(self.param['fftSize']), axis=axis)
        fftframes = np.absolute(fftframes)
        # Do DC reset
        sel_tuple = tuple([(slice(0, fftframes.shape[k]) if k != axis else 0) for k in range(len(fftframes.shape))])
        fftframes[sel_tuple] = 0
        return fftframes

    def filterbank(self, fftframes):
        melframes = np.dot(fftframes, self.full_mel_scale_vec.T)
        melframes = np.where(melframes == 0, np.finfo(float).eps, melframes)
        return melframes

    def logtransform(self, melframes):
        logmelframes = np.log(melframes)
        return logmelframes

    def fe_transform(self, audiodata):
        # Create fixed feature parameters
        self.create_hamming_window()
        self.create_sparse_mel_matrix_part1_part2()
        # Extract the features
        data_dc_removed = self.dc_removal(audiodata)
        frames = self.framing(data_dc_removed)
        fftframes = self.fft(frames)
        melframes = self.filterbank(fftframes)
        logmelframes = self.logtransform(melframes)
        return logmelframes

    def create_plot(self, logmelframes, title):
        fig, ax = plt.subplots()
        ax.imshow(logmelframes.T)
        ax.set_title(title[0:-4])
        plt.xlabel('frames'), plt.ylabel('mel')
        return plt

    def compute_supervec(self, X):
        # Allocate memory
        Xs = np.zeros((np.shape(X)[0], 2*np.shape(X)[2]))
        # Loop over the samples
        for s in range(0, np.shape(X)[0]):
            Xs[s, :] = np.concatenate((np.mean(X[s, :, :], axis=0), np.std(X[s, :, :], axis=0)), axis=0)
        # Return the supervectors
        return Xs

    def normalise_data(self, X):
        if len(np.shape(X)) == 3:
            # Reshape in single matrix
            X_tmp = np.squeeze(np.reshape(X, [1, np.shape(X)[0]*np.shape(X)[1], np.shape(X)[2]]))
            # Compute mean and standard deviation
            mu = np.mean(X_tmp, axis=0)
            sigma = np.std(X_tmp, axis=0)
            # Normalise the data
            for s in range(0, np.shape(X)[0]):
                X[s, :, :] = (X[s, :, :]-mu)/sigma
        elif len(np.shape(X)) == 2:
            # Compute the mean and standard deviation
            mu = np.mean(X, axis=0)
            sigma = np.std(X, axis=0)
            # Normalise the data
            X = (X-mu)/sigma
        # Return the normalised data mean and std
        return X, mu, sigma

    def create_tr_val_te(self, X, Y, ratios):
        if len(np.shape(X)) == 3:
            # Shuffle X and Y
            idx = np.random.permutation(np.shape(X)[0])
            X = X[idx, :, :]
            Y = Y[idx]
            # Define the number of samples per class
            Y_unique, Y_unique_counts = np.unique(Y, return_counts=True)
            # Create empty lists for the data and target values
            X_tr, X_val, X_te, Y_tr, Y_val, Y_te = None, None, None, None, None, None
            # Loop over the classes
            for y_unique in Y_unique:
                # Define the number of training, validation and test elements
                n_tr = np.int(np.floor(Y_unique_counts[y_unique==Y_unique]*ratios[0]))
                n_val = np.int(np.floor(Y_unique_counts[y_unique==Y_unique]*ratios[1]))
                n_te = np.int(Y_unique_counts[y_unique==Y_unique]-n_tr-n_val)
                # search for the indices
                idx = np.where(Y==y_unique)[0]
                # Take the first elements for training
                x_tr = X[idx[0:n_tr], :, :]; y_tr = Y[idx[0:n_tr]]
                x_val = X[idx[n_tr:n_tr+n_val], :, :]; y_val = Y[idx[n_tr:n_tr+n_val]]
                x_te = X[idx[n_tr+n_val:], :, :]; y_te = Y[idx[n_tr+n_val:]]
                # Append to X_tr, X_val, X_te, Y_tr, Y_val, Y_te
                if X_tr is None:
                    X_tr = x_tr; Y_tr = y_tr
                    X_val = x_val; Y_val = y_val
                    X_te = x_te; Y_te = y_te
                else:
                    X_tr = np.append(X_tr, x_tr, axis=0); Y_tr = np.append(Y_tr, y_tr, axis=0)
                    X_val = np.append(X_val, x_val, axis=0); Y_val = np.append(Y_val, y_val, axis=0)
                    X_te = np.append(X_te, x_te, axis=0); Y_te = np.append(Y_te, y_te, axis=0)
        elif len(np.shape(X)) == 2:
            # Shuffle X and Y
            idx = np.random.permutation(np.shape(X)[0])
            X = X[idx, :]
            Y = Y[idx]
            # Define the number of samples per class
            Y_unique, Y_unique_counts = np.unique(Y, return_counts=True)
            # Create empty lists for the data and target values
            X_tr, X_val, X_te, Y_tr, Y_val, Y_te = None, None, None, None, None, None
            # Loop over the classes
            for y_unique in Y_unique:
                # Define the number of training, validation and test elements
                n_tr = np.int(np.floor(Y_unique_counts[y_unique==Y_unique]*ratios[0]))
                n_val = np.int(np.floor(Y_unique_counts[y_unique==Y_unique]*ratios[1]))
                n_te = np.int(Y_unique_counts[y_unique==Y_unique]-n_tr-n_val)
                # search for the indices
                idx = np.where(Y==y_unique)[0]
                # Take the first elements for training
                x_tr = X[idx[0:n_tr], :]; y_tr = Y[idx[0:n_tr]]
                x_val = X[idx[n_tr:n_tr+n_val], :]; y_val = Y[idx[n_tr:n_tr+n_val]]
                x_te = X[idx[n_tr+n_val:], :]; y_te = Y[idx[n_tr+n_val:]]
                # Append to X_tr, X_val, X_te, Y_tr, Y_val, Y_te
                if X_tr is None:
                    X_tr = x_tr; Y_tr = y_tr
                    X_val = x_val; Y_val = y_val
                    X_te = x_te; Y_te = y_te
                else:
                    X_tr = np.append(X_tr, x_tr, axis=0); Y_tr = np.append(Y_tr, y_tr, axis=0)
                    X_val = np.append(X_val, x_val, axis=0); Y_val = np.append(Y_val, y_val, axis=0)
                    X_te = np.append(X_te, x_te, axis=0); Y_te = np.append(Y_te, y_te, axis=0)
        # Return the sets
        return X_tr, X_val, X_te, Y_tr, Y_val, Y_te

    def add_channel_axis(self, X_tr, X_val, X_te):
        # Add axis
        X_tr = X_tr[:, :, :, np.newaxis]
        X_val = X_val[:, :, :, np.newaxis]
        X_te = X_te[:, :, :, np.newaxis]
        # Return the sets
        return X_tr, X_val, X_te
    