import numpy as np
import random
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.fftpack import fft
from collections import defaultdict
#import h5py
import numpy as np
#from sklearn.decomposition import PCA
import scipy 
print(scipy.__version__)
# from scipy import derivative

def standardize_data(arr):
    """
    Standardize the data along the last axis by subtracting the mean and dividing by the standard deviation,
    while safely handling NaN values.

    Parameters:
    arr (numpy.ndarray): Input array, can be 1D, 2D, or higher, with possible NaN values.

    Returns:
    numpy.ndarray: Standardized array with NaNs preserved in their positions.
    """
    isprint= False
    arr = np.asarray(arr)
    
    if np.isneginf(arr).any():
        isprint = True
        pass
    arr = np.where(np.isinf(arr), np.nan, arr)
    # Compute mean and std along the last axis, ignoring NaNs

    if np.isnan(arr).all(axis=-1).all():
        return arr
    #print(np.where(np.isinf(arr), np.nan, arr))
    mean = np.nanmean(arr, axis=-1, keepdims=True)
    std = np.nanstd(arr, axis=-1, keepdims=True)

    # Avoid division by zero (set std to np.nan where it's zero)
    std_safe = np.where(np.isclose(std, 0), 1, std)

    # Standardize while keeping NaNs untouched
    standardized = (arr - mean) / std_safe
    if isprint:
        print(arr)
        print(standardized)
        print('##################################')

    return standardized

class PCA_reduction:

    def __init__(self, n, time = None, signal = None, plot=False):

        if signal is None:
            # Generate example signal (N-point noisy sine wave)
            np.random.seed(42)
            N=1000
            time = np.linspace(0, 10, N)  # Time vector
            signal = np.sin(time) + 0.1 * np.random.randn(N)  # Noisy sine wave

        self.signal = signal
        N = len(signal)
        self.time = time

        # Step 1: Segment the signal into n non-overlapping windows of size  N // num_windows
        num_windows = n  
        window_size = N // num_windows

        X = signal[:num_windows * window_size].reshape(num_windows, window_size)  # Reshape into (N//n, n)

        # Step 2: Apply PCA
        pca = PCA(n_components=1)  # Reduce N//n points to 1 per window
        X_pca = pca.fit_transform(X)  # (n,1) reduced representation

        # Step 3: Extract the 10 key points
        self.reduced_signal = X_pca.flatten()

        # Step 4: Reconstruct the Signal
        X_reconstructed = pca.inverse_transform(X_pca)  # Inverse transform to (N//n, n)
        self.reconstructed_signal = X_reconstructed.flatten()  # Flatten back to N points

        # Step 5: Define new time axis for the reduced points
        reduced_time = np.linspace(time[0], time[-1], num_windows)  # n evenly spaced time points
        if plot or 0:
            # Step 6: Plot Results
            plt.figure(figsize=(12, 5))

            # Original vs. Reduced
            plt.subplot(1, 2, 1)
            plt.plot(time, signal, label="Original Signal", alpha=0.6, linestyle="dashed")
            plt.scatter(reduced_time, self.reduced_signal, color='red', label="PCA Reduced Signal", s=80)
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            plt.title("PCA-Based Signal Reduction (100 → {n} Points)")

            # Original vs. Reconstructed
            plt.subplot(1, 2, 2)
            plt.plot(time, signal, label="Original Signal", alpha=0.6, linestyle="dashed")
            plt.plot(time[:len(self.reconstructed_signal)], self.reconstructed_signal, label="Reconstructed Signal", color='green')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            plt.title("Reconstructed Signal from PCA Components")
            plt.tight_layout()
            plt.show()

def print_structure(name, obj):
    print(name)

class Features:
    ctr = 0
    def __init__(self, time, flow, pressure, area, network_name, artery_name, pca=None, density=1050):
        """
        Initialize the Features class with time-series data.
        :param time: Time array (seconds)
        :param flow: Flow rate array (mL/s or cm³/s)
        :param pressure: Pressure array (mmHg)
        :param area: Vessel cross-sectional area array (mm²)
        :param density: Blood density (kg/m³), default is 1050 kg/m³.
        """
        #self.ctr = 0
        self.time = time
        self.flow = flow
        self.pressure = pressure
        self.area = area
        self.density = density  # Blood density for impedance calculation
        self.dt = np.mean(np.diff(time))  # Time step
        self.network_name = network_name
        self.artery_name = artery_name
        self.validity = True
        self.pca = pca
    # FLOW
    def peak_systolic_flow(self):
        #check
        return np.max(self.flow)
    
    def mean_flow_rate(self):
        #check
        return np.mean(self.flow)
    
    def pulsatility_index(self):
        #check
        return (np.max(self.flow) - np.min(self.flow)) / np.mean(self.flow)
    
    def max_flow_acceleration(self):
        # check
        return np.max(np.gradient(self.flow, self.dt))
    
    def fourier_coefficients(self, n=6):
        # check
        """Compute first n Fourier coefficients of the flow waveform."""
        fft_vals = fft(self.flow)
        return np.abs(fft_vals[:n])
    
    def flow_PCA(self):
        pca_signal = PCA_reduction(signal=self.flow, n=self.pca, time=self.time)
        
        return pca_signal.reduced_signal
    
    # PRESSURE
    def systolic_pressure(self):
        # check
        return np.max(self.pressure)
    
    def diastolic_pressure(self):
        # check
        return np.min(self.pressure)
    
    def pulse_pressure(self):#PP
        # check
        return self.systolic_pressure() - self.diastolic_pressure()
    
    def augmentation_index(self):
        # check
        self.peaks, _ = find_peaks(self.pressure)
        if len(self.peaks) > 1:
            return 100*(self.pressure[self.peaks[1]] - self.diastolic_pressure()) / self.pulse_pressure()
        
        # print(self.network_name, ' ' , self.artery_name)
        self.validity = False
        self.ctr=self.ctr + 1
        global kkk
        global bad_networks
        bad_networks.append(self.network_name)
        kkk+=1
        #print(kkk)
        if 0:
            plt.figure()
            plt.title(self.artery_name)
            plt.plot(self.time, self.pressure)
            plt.show()
        return -np.inf
    
    def wave_reflection_time(self):
        # check
        if not hasattr(self, 'peaks'):
            self.peaks, _ = find_peaks(self.pressure)
        if len(self.peaks) > 1:
            return self.time[self.peaks[1]] - self.time[self.peaks[0]]
        #print(self.name)
        return -np.inf
    
    def pressure_rise_time(self):
        # check
        # Time from diastole to peak systolic pressure
        # first ccurence of min
        return self.time[np.argmax(self.pressure)] - self.time[np.where(self.pressure == np.min(self.pressure))[0][0]]
    
    def pressure_PCA(self):
        #pca = PCA(n_components=n)
        pca_signal = PCA_reduction(signal=self.pressure, n=self.pca, time=self.time)
        return pca_signal.reduced_signal
    # Area
    def min_area(self):
        # check
        return np.min(self.area)
    
    def max_area(self):
        # check
        return np.max(self.area)
    
    def area_compliance_index(self):
        # check
        return (self.max_area() - self.min_area()) / self.pulse_pressure()
    
    def distensibility_coefficient(self):
        # check
        return ((self.max_area() - self.min_area()) / self.min_area()) / self.pulse_pressure()
    
    def wave_speed(self):
        #check
        """Estimate pulse wave velocity (PWV) using Moens-Korteweg equation."""
        return np.sqrt(self.pulse_pressure() / (self.density * self.area_compliance_index()))
    
    def characteristic_impedance(self):
        #check
        """Estimate characteristic impedance."""
        return self.density * self.wave_speed() / np.mean(self.area)
    
    def total_arterial_compliance(self):
        #check
        """Estimate total arterial compliance using Windkessel-like model."""
        return 1 / (self.characteristic_impedance() * self.wave_speed())
    
    def extract_all_features(self):
        """Extract all relevant features into a dictionary."""
        return {
            'Q_peak': self.peak_systolic_flow(),
            'Q_mean': self.mean_flow_rate(),
            'Pulsatility_Index': self.pulsatility_index(),
            'Flow_Acceleration': self.max_flow_acceleration(),
            'Fourier_Coeffs': self.fourier_coefficients(),
            'P_sys': self.systolic_pressure(),
            'P_dia': self.diastolic_pressure(),
            'Pulse_Pressure': self.pulse_pressure(),
            'Augmentation_Index': self.augmentation_index(),
            'Wave_Reflection_Time': self.wave_reflection_time(),
            'Pressure_Rise_Time': self.pressure_rise_time(),
            'A_min': self.min_area(),
            'A_max': self.max_area(),
            'Area_Compliance_Index': self.area_compliance_index(),
            'Distensibility_Coefficient': self.distensibility_coefficient(),
            'Wave_Speed': self.wave_speed(),
            'Characteristic_Impedance': self.characteristic_impedance(),
            'Total_Arterial_Compliance': self.total_arterial_compliance(),
            'Pressure_PCA': self.pressure_PCA(),
            'Flow_PCA': self.flow_PCA()
        }, self.validity

def create_batch(hdf5_file = 'artery_networks.h5'):
    # Load the HDF5 file
    with h5py.File(hdf5_file, 'r') as f:
        data = []
        # Initialize lists to hold the data
        # f.visititems(print_structure)
         
        # Iterate over the experiments in the HDF5 file
        for exp_name in f.keys():
            exp = f[exp_name]
            network = {} # defaultdict()
            network['name'] = exp_name
            network['network'] = {}
            # Iterate over the features in each experiment
            for artery_name in exp:
                artery = exp[artery_name]
                network['network'][artery_name] = {}
                # Iterate over the signals in each feature
                for signal_name in artery:
                    
                    if len(artery[signal_name].shape):
                        signal_data = artery[signal_name][:]
                    else:
                        signal_data = artery[signal_name][()]
                    network['network'][artery_name][signal_name] = signal_data
                    
                    # Add the signal 
            data.append(network)
    return data


if __name__ == "__main__":
    if 0:
        my_files = open("experiments.txt", "r") 
        
        # reading the file 
        data = my_files.read() 
        
        # replacing end of line('/n') with ' ' and 
        # splitting the text it further when '.' is seen. 
        data_into_list = data.split("\n") 
        data_into_list = data_into_list[:-1]
        # printing the data 
        print(data_into_list) 
        # plot some simulation results
    if 1:
        #model1, model2 = random.sample(data_into_list, 2)

        #my_files.close() 
        dir_list = os.listdir('/home/ahmed/Desktop/projects/project7/adan56_results/')      #'results/'+model1+'_results')
        artery = random.sample(dir_list, 1)[0]
        plt.figure()
        for artery in dir_list:
            if artery[-6]!='P':
                continue
            all_artery = np.empty((100, 0))
            #for model in data_into_list:
            data1 = np.loadtxt('adan56_results/'+artery)
            all_artery = np.hstack((all_artery, data1[:, [-1]]))
            # data2 = np.loadtxt('results/'+model2+'_results/'+artery)
            
            #plt.title(artery)
            plt.plot(all_artery)
        plt.show()

    # test Features class

    # build dict for taining from result folder
    if 0:
        data = []
        for model_name in data_into_list: #models#
            model_data = os.listdir('results/'+model_name+'_results')
            network = {} # defaultdict()
            network['name'] = model_name
            network['network'] = {}
            for artery_signal in model_data: #arteries:
                if 'last' not in artery_signal:
                    continue
                artery, signal = artery_signal[:-7], artery_signal[-6]
                network['network'].setdefault(artery,{})[signal] =  np.loadtxt('results/' + model_name + '_results/' + artery_signal)
            data.append(network)
    if 0:
        # Create an HDF5 file to save data from the saved data 
        with h5py.File('artery_networks.h5', 'w') as f:
            for model_name in data_into_list: #models#
                model_data = os.listdir('results/' + model_name + '_results')
                # Create an experiment group
                exp = f.create_group(model_name)
                
                # Create a features group for each experiment
                #network_group = exp_group.create_group('network')
                
                for artery_signal in model_data: #arteries:
                    if 'last' not in artery_signal:
                        continue
                    artery_name, signal_name = artery_signal[:-7], artery_signal[-6]
                    if artery_name not in exp :
                        artery = exp.create_group(artery_name)
                        #artery.create_group(artery_name)
                        #artery_group['name'] = artery
                    else:
                        artery = exp[artery_name]
                    
                    #for signal_name, signal_data in signals.items():
                        # Store the signal data in the feature group
                    
                    artery.create_dataset(f'{signal_name}', data = np.loadtxt('results/' + model_name + '_results/' + artery_signal))


    data = create_batch(hdf5_file = 'artery_networks.h5')
    T = .917
    global kkk
    global bad_networks
    bad_networks = []
    kkk = 0
    mesh = [ 1, 2, 3, 4, 5] # index 0 for t
    Networks_features = {}#list()#np.empty((100, 6))
    for network in data:
        Network_features = {}
        Network_features['name'] = network['name']
        Network_features['network'] = {}
        validity = True
        time = np.linspace(0, T, 100) # len(artery['Q']))
        merged_dict = {}
        for artery_name, artery in network['network'].items():
            merged_dict[artery_name] = {} # defaultdict(list)
            if 0:    
                plt.figure()
                plt.subplot(1,2,1)
                plt.title('pressure')
                plt.plot(network['network']['aortic_arch_IV']['P'][:, -1], label='ground truth')
                plt.plot(network['network']['aortic_arch_I']['P'][:, -1], label='fitted')
                plt.legend()
                
                plt.subplot(1,2,2)
                plt.plot(network['network']['aortic_arch_IV']['Q'][:, -1], label='ground truth')
                plt.plot(network['network']['aortic_arch_I']['Q'][:, -1], label='fitted')
                plt.legend()
                
                plt.title('flow')
                plt.show()
            
            if 0:
                plt.figure()
                plt.subplot(1,2,1)
                plt.title('pressure')
                plt.plot(artery['P'][:, -1])
                plt.subplot(1,2,2)
                plt.plot(artery['Q'][:, -1])
                plt.title('flow')
                plt.show()
            Network_features['network'][artery_name] = []
            if 0:

                dp = np.gradient(artery['P'][:, 0], time[1]-time[0], edge_order=2)#, time)
                ddp = np.gradient(dp, time[1]-time[0], edge_order=2)#, time)

                plt.figure()
                plt.ylabel('P')
                plt.plot(time, artery['P'][:, 0])
                plt.figure()
                plt.ylabel('dP')
                plt.plot(time, dp)
                plt.figure()
                plt.ylabel('ddP')
                plt.plot(time, ddp)
                plt.show()

            for x in mesh:
                if 0:
                    plt.figure()
                    plt.title(artery_name+'  '+str(x))
                    plt.subplot(3, 1, 1)
                    plt.ylabel('Q')
                    plt.plot(time, artery['Q'][:, x])
                    plt.subplot(3, 1, 2)
                    plt.ylabel('P')
                    plt.plot(time, artery['P'][:, x])
                    plt.subplot(3, 1, 3)
                    plt.ylabel('A')
                    plt.plot(time, artery['A'][:, x])
                    plt.show()
                signal, validity = Features(time, artery['Q'][:, x], artery['P'][:, x], artery['A'][:, x], network['name'], artery_name, pca=20).extract_all_features()
                if not validity:
                    pass
                if not validity and 0:
                    break
                Network_features['network'][artery_name].append(signal)
                if 0:
                    ss = []
                    for s in signal.values():
                        
                        if isinstance(s, np.ndarray):
                            for tt in s.flatten():
                                ss.append(tt )
                        else:
                            ss.append(s)
                    plt.figure()
                    plt.plot(ss)
                    plt.show()
            if not validity and 0:
                break
            if Network_features['network'][artery_name]:  # Ensure there is at least one array
                for d in Network_features['network'][artery_name]:
                    for key, value in d.items():
                        if key in merged_dict[artery_name]:
                            merged_dict[artery_name][key].append(value)
                        else:
                            merged_dict[artery_name][key] = [value]
            pass
        if validity or 1:
            if 1:
                for artery, value in merged_dict.items():
                    for feature, signal in value.items():

                        merged_dict[artery][feature] = standardize_data(merged_dict[artery][feature])
            Networks_features[network['name']] = merged_dict

    pass

    # Create an HDF5 file to save data from the saved data 
    with h5py.File('artery_networks_features_not_clean_standrized_std_july.h5', 'w') as f:
        for model_name, model_features in Networks_features.items(): #models#
            #model_data = os.listdir('results/' + model_name + '_results')
            # Create an experiment group
            exp = f.create_group(model_name)
            # Create a features group for each experiment
            #network_group = exp_group.create_group('network')
            
            for artery_name, artery_features in model_features.items(): #arteries:

                if artery_name not in exp :
                    artery = exp.create_group(artery_name)
                    #artery.create_group(artery_name)
                    #artery_group['name'] = artery
                else:
                    artery = exp[artery_name]
                
                #for signal_name, signal_data in signals.items():
                    # Store the signal data in the feature group
                
                #artery.create_group(artery_name)
                for feature, value in artery_features.items():
                    try:
                        artery.create_dataset(feature, data = value)
                    except:
                        print(value)
                        pass

    with h5py.File('artery_networks_features_clean_log_nofourrier.h5', 'r') as f:              
        pass

    data = create_batch(hdf5_file = 'artery_networks_features_clean.h5')
    pass
    # Optionally, you can encode the labels into numeric form if needed for training
    # Example: If y contains feature names, you might want to encode them numerically
    # from sklearn.preprocessing import LabelEncoder
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)

