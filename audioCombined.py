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
    return magnitude_spectrogram_denoised, phase_spectrogram

# Function for Wiener Filtering
def wiener_filtering(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed):
    signal_power = magnitude_spectrogram**2
    noise_power = noise_spectrum_smoothed[:, np.newaxis]**2
    wiener_filter = signal_power / (signal_power + noise_power)
    magnitude_spectrogram_denoised = magnitude_spectrogram * wiener_filter
    return magnitude_spectrogram_denoised, phase_spectrogram

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

# Function to calculate SNR
def calculate_snr(original, denoised):
    # Ensure the same length for both signals
    min_len = min(len(original), len(denoised))
    original = original[:min_len]
    denoised = denoised[:min_len]
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - denoised) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to reconstruct noise signal from estimated noise spectrum
def reconstruct_noise_signal(noise_spectrum_smoothed, length, n_fft, hop_length):
    noise_stft = np.tile(noise_spectrum_smoothed[:, np.newaxis], (1, length // hop_length + 1))
    noise_signal = librosa.istft(noise_stft * np.exp(1j * np.random.uniform(0, 2*np.pi, noise_stft.shape)),
                                 hop_length=hop_length, length=length)
    return noise_signal

# Function to plot and save noise in time domain
def plot_noise_time_domain(noise_signal, sr):
    time = np.linspace(0, len(noise_signal) / sr, len(noise_signal))
    plt.figure()
    plt.plot(time, noise_signal)
    plt.title('Estimated Noise in Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('estimated_noise_time_domain.png')
    plt.show()

# Main function to apply the chosen noise reduction method
def apply_combined_noise_reduction(file_name, noise_estimation=''):
    # Load the audio file with a sample rate of 32000 Hz
    y, sr = librosa.load(f'{file_name}.wav', sr=32000)

    # Display and save the original waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, axis='s', ax=ax)
    ax.set_title(f'Waveform of {file_name}.wav')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.savefig('original_waveform.png')
    plt.show()

    # Perform STFT (Short-Time Fourier Transform)
    n_fft = 2048
    hop_length = 256
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft_result)
    phase_spectrogram = np.angle(stft_result)

    # Estimate noise spectrum
    if noise_estimation == 'manual':
        start_time = 2.3
        end_time = 2.8
        noise_spectrum_smoothed = estimate_noise_manual(y, sr, hop_length, n_fft, start_time, end_time)
    elif noise_estimation == 'low_energy':
        noise_spectrum_smoothed = estimate_noise_low_energy(y, sr, hop_length, n_fft)
    else:
        raise ValueError("Invalid noise estimation method. Choose 'manual' or 'low_energy'.")

    # Reconstruct and plot the noise signal
    noise_signal = reconstruct_noise_signal(noise_spectrum_smoothed, len(y), n_fft, hop_length)
    plot_noise_time_domain(noise_signal, sr)

    # Step 1: Spectral Subtraction
    magnitude_spectrogram_subtracted, phase_spectrogram_subtracted = spectral_subtraction(magnitude_spectrogram, phase_spectrogram, noise_spectrum_smoothed)

    # Step 2: Wiener Filtering
    magnitude_spectrogram_denoised, phase_spectrogram_denoised = wiener_filtering(magnitude_spectrogram_subtracted, phase_spectrogram_subtracted, noise_spectrum_smoothed)

    # Reconstruct the signal from the denoised magnitude spectrogram
    stft_denoised = magnitude_spectrogram_denoised * np.exp(1j * phase_spectrogram_denoised)
    y_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

    # Output Processing: Save the cleaned audio to a file
    sf.write(f'{file_name}_denoised_combined.wav', y_denoised, sr)

    # Plot the denoised spectrogram
    plot_spectrogram(np.abs(stft_denoised), f'Denoised Spectrogram (Combined) - {file_name}', sr, hop_length)

    # Plot the original and denoised waveforms for comparison and save the plot
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, label='Original')
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax, alpha=0.75, label='Denoised (Combined)')
    ax.set_title(f'Waveform Comparison (Combined) - {file_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.savefig('waveform_comparison_combined.png')
    plt.show()

    # Print some details about the processing
    print(f"Original audio shape: {y.shape}")
    y_denoised, _ = sf.read(f'{file_name}_denoised_combined.wav')
    print(f"Denoised audio shape (Combined): {y_denoised.shape}")

    # Calculate and print SNR
    snr = calculate_snr(y, y_denoised)
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

# Call the main function with the desired file name and noise estimation method
apply_combined_noise_reduction('h_3', noise_estimation='low_energy')  # Change to 'h_1', 'h_2', or 'h_3' and 'manual/low_energy' as needed
