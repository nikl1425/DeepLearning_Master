from scipy import signal


def butter_highpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    y = signal.filtfilt(b, a, data)
    return y


def apply_filter(series, FREQ, low=False):
    series = butter_bandstop_filter(series, 97, 103, FREQ, order=4)
    series = butter_bandstop_filter(series, 47, 53, FREQ, order=4)
    series = butter_highpass_filter(series, 1, FREQ, order=4)
    if low:
        series = butter_lowpass_filter(series, 240, FREQ, order=3)
    return series


def apply_ecg_filter(series, FREQ):
    series = butter_bandstop_filter(series, 47, 53, FREQ, order=4)
    series = butter_highpass_filter(series, 1, FREQ, order=4)
    series = butter_bandpass_filter(series, 1, 98, FREQ)
    return series
