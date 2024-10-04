import numpy as np
import matplotlib.pyplot as plt

# Paramètres du signal
N = 1000  # Nombre de points d'échantillonnage
T = 1.0 / 800.0  # Intervalle d'échantillonnage (en secondes)
t = np.linspace(0.0, N*T, N, endpoint=False)  # Vecteur temps
freq = 5  # Fréquence du signal (en Hz)
amplitude = 1.0  # Amplitude du signal

# Génération du signal sinusoïdal
signal = amplitude * np.sin(2.0 * np.pi * freq * t)

# Ajout de bruit au signal
noise_amplitude = 0.5
noise = noise_amplitude * np.random.normal(size=t.shape)
noisy_signal = signal + noise

# Calcul de la DFT
dft = np.fft.fft(noisy_signal)
dft_magnitude = np.abs(dft)  # Magnitude de la DFT
dft_phase = np.angle(dft)  # Phase de la DFT

# Fréquences correspondantes
frequencies = np.fft.fftfreq(N, T)

# Filtrage fréquentiel (exemple de filtre passe-bas)
cutoff_freq = 10  # Fréquence de coupure (en Hz)
dft_filtered = dft.copy()
dft_filtered[np.abs(frequencies) > cutoff_freq] = 0  # Suppression des hautes fréquences

# Signal reconstruit après filtrage
filtered_signal = np.fft.ifft(dft_filtered)

# Visualisation
plt.figure(figsize=(18, 12))

# Signal temporel original
plt.subplot(321)
plt.plot(t, signal)
plt.title('Signal temporel original')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')

# Signal temporel avec bruit
plt.subplot(322)
plt.plot(t, noisy_signal)
plt.title('Signal temporel avec bruit')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')

# Magnitude de la DFT
plt.subplot(323)
plt.plot(frequencies, dft_magnitude)
plt.title('Magnitude de la DFT')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Magnitude')

# Phase de la DFT
plt.subplot(324)
plt.plot(frequencies, dft_phase)
plt.title('Phase de la DFT')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Phase [radians]')

# Magnitude de la DFT après filtrage
plt.subplot(325)
plt.plot(frequencies, np.abs(dft_filtered))
plt.title('Magnitude de la DFT après filtrage')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Magnitude')

# Signal temporel reconstruit après filtrage
plt.subplot(326)
plt.plot(t, filtered_signal.real)
plt.title('Signal temporel reconstruit après filtrage')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
