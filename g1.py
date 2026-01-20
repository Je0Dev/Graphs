import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.interpolate import interp1d

# --- 1. Ορισμός Σήματος και Σταθερών ---

# Παράμετροι σήματος
f1 = 400   # Hz
f2 = 800   # Hz
f3 = 1200  # Hz
f_max = 1200 # Hz
f_nyquist = 2 * f_max

# Παράμετροι χρόνου και δειγματοληψίας
T_duration = 0.02  # Διάρκεια σήματος (0.02s αρκεί για να δούμε μερικές περιόδους)
fs_cont = 50 * f_max # "Συνεχής" προσομοίωση (υψηλός ρυθμός)
fs1 = 4000         # Ρυθμός Nyquist (Άσκηση α)
fs2 = 1500         # Ρυθμός Aliasing (Άσκηση β)

# Χρονικοί άξονες
# "Συνεχής" χρόνος για το αρχικό σήμα
t_cont = np.linspace(0, T_duration, int(fs_cont * T_duration), endpoint=False)
# Χρόνος δειγματοληψίας για fs1
t_samp1 = np.arange(0, T_duration, 1/fs1)
# Χρόνος δειγματοληψίας για fs2
t_samp2 = np.arange(0, T_duration, 1/fs2)

# Συνάρτηση ορισμού σήματος
def m(t):
    return 2 * np.cos(2 * np.pi * f1 * t) + \
           np.cos(2 * np.pi * f2 * t) - \
           3 * np.sin(2 * np.pi * f3 * t)

# Δημιουργία σημάτων
m_cont = m(t_cont)
m_samp1 = m(t_samp1)
m_samp2 = m(t_samp2)

print(f"f_max = {f_max} Hz, f_Nyquist = {f_nyquist} Hz")
print(f"fs1 = {fs1} Hz (No Aliasing)")
print(f"fs2 = {fs2} Hz (Aliasing)")

# --- (a) Δειγματοληψία με 4000 Hz και Φάσματα ---

print("\n--- Μέρος (a): Δειγματοληψία & Φάσματα (fs=4000 Hz) ---")

# Συνάρτηση για υπολογισμό FFT
def get_spectrum(signal, fs):
    N = len(signal)
    # Χρησιμοποιούμε norm='ortho' για να διατηρείται η ισχύς
    M_f = fftshift(fft(signal, n=N, norm='ortho'))
    freq_axis = fftshift(fftfreq(N, 1/fs))
    return freq_axis, np.abs(M_f)

# Υπολογισμός φασμάτων
freq_cont, mag_cont = get_spectrum(m_cont, fs_cont)
freq_samp1, mag_samp1 = get_spectrum(m_samp1, fs1)

# Σχεδίαση
plt.figure(figsize=(14, 10))

# 1. Αρχικό σήμα στο χρόνο
plt.subplot(2, 2, 1)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$ (Προσομοίωση)')
plt.stem(t_samp1, m_samp1, 'r', markerfmt='ro', basefmt='r-', 
         label=f'Δείγματα $m[n]$ ($f_s={fs1}$ Hz)')
plt.title(f'(a) Σήμα και Δείγματα ($f_s={fs1}$ Hz)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)

# 2. Φάσμα αρχικού σήματος
plt.subplot(2, 2, 2)
plt.plot(freq_cont, mag_cont, 'b')
plt.title('(a) Φάσμα Αρχικού Σήματος |M(f)|')
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Μέτρο')
plt.xlim(-fs1/2, fs1/2) # Εστίαση στη βασική ζώνη
plt.grid(True)

# 3. Φάσμα δειγματοληπτημένου σήματος
plt.subplot(2, 2, (3, 4))
plt.plot(freq_samp1, mag_samp1, 'r-')
plt.title(f'(a) Φάσμα Δειγματοληπτημένου Σήματος ($f_s={fs1}$ Hz)')
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Μέτρο')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- (b) Ανακατασκευή και Aliasing (fs=1500 Hz) ---

print("\n--- Μέρος (b): Ανακατασκευή & Aliasing ---")

# Συνάρτηση για ιδανική ανακατασκευή LPF
def ideal_lpf_reconstruct(signal_samples, t_samples, t_target, fs_sample):
    # Προσθήκη padding για καλύτερη απόκριση FFT
    N_pad = len(t_target) 
    
    # Μετασχηματισμός Fourier των δειγμάτων
    M_f = fft(signal_samples, n=N_pad)
    freq_axis = fftfreq(N_pad, 1/fs_sample)
    
    # Εφαρμογή Ιδανικού LPF
    # Αποκοπή συχνοτήτων πάνω από fs_sample / 2 (Nyquist freq)
    B_lpf = fs_sample / 2
    M_f[np.abs(freq_axis) > B_lpf] = 0
    
    # Αντίστροφος FFT για ανακατασκευή
    # Πολλαπλασιάζουμε με N_pad / len(signal_samples) για σωστή κλιμάκωση
    # λόγω του padding και του FFT
    scaling_factor = N_pad / len(signal_samples)
    m_recon = ifft(M_f, n=N_pad) * scaling_factor
    
    # Επιστροφή μόνο του πραγματικού μέρους
    return np.real(m_recon)

# Σενάριο 1: fs = 4000 Hz, B = 2000 Hz
m_recon_fs1 = ideal_lpf_reconstruct(m_samp1, t_samp1, t_cont, fs1)
# Υπολογισμός σφάλματος
error_fs1 = m_cont - m_recon_fs1

# Σενάριο 2: fs = 1500 Hz, B = 750 Hz (Aliasing)
# Χρειαζόμαστε t_target που να ταιριάζει με το t_cont
m_recon_fs2 = ideal_lpf_reconstruct(m_samp2, t_samp2, t_cont, fs2)
# Υπολογισμός σφάλματος
error_fs2 = m_cont - m_recon_fs2

# Σχεδίαση
plt.figure(figsize=(14, 10))

# 1. Τέλεια Ανακατασκευή (fs=4000)
plt.subplot(2, 2, 1)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_cont, m_recon_fs1, 'g--', label='Ανακατασκευή ($f_s=4000$ Hz)')
plt.title(f'(b) Τέλεια Ανακατασκευή ($f_s={fs1}$ Hz, $B={fs1/2}$ Hz)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)

# 2. Σφάλμα Τέλειας Ανακατασκευής
plt.subplot(2, 2, 2)
plt.plot(t_cont, error_fs1, 'r-')
plt.title('Σφάλμα (Διαφορά) - (Ιδανικά Μηδέν)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Σφάλμα')
plt.grid(True)

# 3. Ανακατασκευή με Aliasing (fs=1500)
plt.subplot(2, 2, 3)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_cont, m_recon_fs2, 'r--', label='Ανακατασκευή ($f_s=1500$ Hz)')
plt.title(f'(b) Ανακατασκευή με Aliasing ($f_s={fs2}$ Hz, $B={fs2/2}$ Hz)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)

# 4. Σφάλμα Aliasing
plt.subplot(2, 2, 4)
plt.plot(t_cont, error_fs2, 'r-')
plt.title('Σφάλμα (Διαφορά) - (Μη Μηδενικό λόγω Aliasing)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Σφάλμα')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- (c) Ανακατασκευή με ZOH και Equalizer (fs=4000 Hz) ---

print("\n--- Μέρος (c): ZOH & Equalizer (fs=4000 Hz) ---")

Ts1 = 1/fs1

# Προσομοίωση ανακατασκευής ZOH χρησιμοποιώντας παρεμβολή 'zero'
# (γνωστή και ως 'previous' ή 'step')
f_zoh = interp1d(t_samp1, m_samp1, kind='previous', fill_value='extrapolate')
m_recon_zoh = f_zoh(t_cont)

# Σχεδιασμός Φίλτρου Equalizer
N_cont = len(t_cont)
freq_axis_cont = fftshift(fftfreq(N_cont, 1/fs_cont))

# Απόκριση ZOH
# Χρησιμοποιούμε epsilon για αποφυγή διαίρεσης με το 0 στο H_eq
epsilon = 1e-9
H_zoh = Ts1 * np.sinc(freq_axis_cont * Ts1) * np.exp(-1j * np.pi * freq_axis_cont * Ts1)

# Ιδανικό LPF (ως μάσκα για τον equalizer)
B_lpf = fs1 / 2
H_lpf = np.where(np.abs(freq_axis_cont) <= B_lpf, 1, 0)

# Φίλτρο Equalizer (Inverse Sinc)
H_eq = H_lpf / (H_zoh + epsilon)

# Εφαρμογή του Equalizer
# 1. Παίρνουμε το φάσμα του σήματος ZOH
M_recon_zoh_f = fftshift(fft(m_recon_zoh, n=N_cont))
# 2. Εφαρμόζουμε το φίλτρο equalizer
M_equalized_f = M_recon_zoh_f * H_eq
# 3. Αντίστροφος FFT
m_equalized = np.real(ifft(ifftshift(M_equalized_f)))

# Σχεδίαση
plt.figure(figsize=(14, 12))

# 1. Σύγκριση ZOH με Αρχικό
plt.subplot(3, 1, 1)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$', alpha=0.7)
plt.plot(t_cont, m_recon_zoh, 'r--', label='Ανακατασκευή ZOH ("Σκαλοπάτια")')
plt.title('(c) Ανακατασκευή με Zero-Order Hold (ZOH)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)

# 2. Απόκριση Φίλτρων (ZOH και Equalizer)
plt.subplot(3, 1, 2)
plt.plot(freq_axis_cont, np.abs(H_zoh), 'g-', label='$|H_{ZOH}(f)|$ (Sinc Droop)')
plt.plot(freq_axis_cont, np.abs(H_eq), 'm-', label='$|H_{EQ}(f)|$ (Inverse Sinc)')
plt.title('(c) Απόκριση Συχνότητας Φίλτρων ZOH και Equalizer')
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('Μέτρο')
plt.xlim(-fs1, fs1)
plt.ylim(0, 5) # Περικοπή για να φαίνεται η ενίσχυση
plt.legend()
plt.grid(True)

# 3. Σύγκριση Εξισορροπημένου Σήματος
plt.subplot(3, 1, 3)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_cont, m_equalized, 'm--', label='ZOH + Equalizer')
plt.title('(c) Αποτέλεσμα μετά την Εξισορρόπηση')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
