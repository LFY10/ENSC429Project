import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import medfilt

# Function to plot spectrogram
def plot_spectrogram(magnitude_spectrogram, title, sr, hop_length):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max),
                                   sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

# Function for Spectral Subtraction
def spectral_subtraction(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed):
    magnitude_spectrogram_denoised = np.maximum(magnitude_spectrogram - noise_spectrum_smoothed[:, np.newaxis], 0)
    stft_denoised = magnitude_spectrogram_denoised * np.exp(1j * phase_spectrogram)
    return stft_denoised

# Function for Wiener Filtering
def wiener_filtering(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed):
    signal_power = magnitude_spectrogram**2
    noise_power = noise_spectrum_smoothed[:, np.newaxis]**2
    wiener_filter = signal_power / (signal_power + noise_power)
    magnitude_spectrogram_denoised = magnitude_spectrogram * wiener_filter
    stft_denoised = magnitude_spectrogram_denoised * np.exp(1j * phase_spectrogram)
    return stft_denoised



# Function to estimate noise from low energy segments
def estimate_noise_low_energy(y, sr, hop_length, n_fft):
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft_result)
    energy = np.sum(magnitude_spectrogram ** 2, axis=0)
    low_energy_frames = np.argsort(energy)[:int(len(energy) * 0.1)]  # Use the lowest 10% energy frames
    noise_spectrogram = magnitude_spectrogram[:, low_energy_frames]
    noise_spectrum = np.mean(noise_spectrogram, axis=1)
    noise_spectrum_smoothed = medfilt(noise_spectrum, kernel_size=5)
    return noise_spectrum_smoothed

# Function to estimate noise from a manually specified segment
def estimate_noise_manual(y, sr, hop_length, n_fft, start_time, end_time):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    noise_segment = y[start_sample:end_sample]
    
    stft_noise = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_spectrum = np.mean(np.abs(stft_noise), axis=1)
    noise_spectrum_smoothed = medfilt(noise_spectrum, kernel_size=5)
    return noise_spectrum_smoothed

# Main function to apply the chosen noise reduction method
def apply_noise_reduction(noise_estimation='', method=''):
    # Load the audio file 'h_1.wav' with a sample rate of 32000 Hz
    y, sr = librosa.load('h_1.wav', sr=32000)

    # Display the original waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, axis='s', ax=ax)
    ax.set_title('Waveform of h_1.wav')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.savefig('original_waveform.png')
    plt.show()

    # Perform STFT (Short-Time Fourier Transform)
    n_fft = 1024
    hop_length = 512
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft_result)
    phase_spectrogram = np.angle(stft_result)

    # Estimate noise spectrum
    if noise_estimation == 'manual':
        start_time = 2.3  #Noise duration from the plot
        end_time = 2.9
        noise_spectrum_smoothed = estimate_noise_manual(y, sr, hop_length, n_fft, start_time, end_time)
    elif noise_estimation == 'low_energy':
        noise_spectrum_smoothed = estimate_noise_low_energy(y, sr, hop_length, n_fft)
    else:
        raise ValueError("Invalid noise estimation method.")

    # Plot the estimated noise spectrum
    fig, ax = plt.subplots()
    ax.plot(noise_spectrum_smoothed)
    ax.set_title(f'Estimated Noise Spectrum ({noise_estimation.upper()})')
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Magnitude')
    plt.show()

    # Plot the original spectrogram
    plot_spectrogram(magnitude_spectrogram, 'Original Spectrogram', sr, hop_length)

    # Apply the selected noise reduction method and plot results
    if method == 'spectral_subtraction':
        stft_denoised = spectral_subtraction(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed)
    elif method == 'wiener':
        stft_denoised = wiener_filtering(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed)
    else:
        raise ValueError("Invalid method. Choose 'spectral_subtraction' or 'wiener'.")

    y_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

    # Output Processing: Save the cleaned audio to a file (optional)
    sf.write(f'h_1_denoised_{noise_estimation}_{method}.wav', y_denoised, sr)

    # Plot the denoised spectrogram
    plot_spectrogram(np.abs(stft_denoised), f'Denoised Spectrogram ({noise_estimation.upper()}, {method.capitalize()})', sr, hop_length)

    # Plot the original and denoised waveforms for comparison and save the plot
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, label='Original')
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax, alpha=0.75, label=f'Denoised ({method.capitalize()})')
    ax.set_title(f'Waveform Comparison ({noise_estimation.upper()}, {method.capitalize()})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.savefig(f'waveform_comparison_{noise_estimation}_{method}.png')
    plt.show()

    # Print some details about the processing
    print("Original audio shape: ", y.shape)
    y_denoised, _ = sf.read(f'h_1_denoised_{noise_estimation}_{method}.wav')
    print(f"Denoised audio shape ({noise_estimation.upper()}, {method.capitalize()}): ", y_denoised.shape)

# Change noise_estimation: 'manual/low_energy' or denoised_method:'wiener/spectral_subtraction'
apply_noise_reduction(noise_estimation='manual', method='wiener')  
