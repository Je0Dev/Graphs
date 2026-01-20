import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.interpolate import interp1d

# --- 1. Ορισμός Σήματος και Σταθερών (από 5.8-1) ---

f1 = 400; f2 = 800; f3 = 1200; f_max = 1200;
T_duration = 0.02
fs_cont = 50 * f_max
fs1 = 4000
Ts1 = 1/fs1

t_cont = np.linspace(0, T_duration, int(fs_cont * T_duration), endpoint=False)
t_samp1 = np.arange(0, T_duration, Ts1)

def m(t):
    return 2 * np.cos(2 * np.pi * f1 * t) + \
           np.cos(2 * np.pi * f2 * t) - \
           3 * np.sin(2 * np.pi * f3 * t)

m_cont = m(t_cont)
m_samp1 = m(t_samp1)

# --- 2. Συνάρτηση Ομοιόμορφου Κβαντιστή ---

def quantize_uniform(signal, L, v_min, v_max):
    """
    Κβαντίζει ένα σήμα 'signal' σε L επίπεδα,
    στη δυναμική περιοχή [v_min, v_max].
    """
    delta = (v_max - v_min) / L
    
    # 1. Μετατόπιση στην περιοχή [0, L*delta]
    signal_shifted = signal - v_min
    
    # 2. Εύρεση του "δείκτη" επιπέδου (0 έως L-1)
    indices = np.floor(signal_shifted / delta)
    
    # 3. Περιορισμός (clipping) στις άκρες [0, L-1]
    indices = np.clip(indices, 0, L - 1)
    
    # 4. Υπολογισμός κβαντισμένης τιμής (κέντρο του επιπέδου)
    quantized_values = v_min + (indices + 0.5) * delta
    
    return quantized_values

# --- 3. Κβάντιση Σήματος ---

L_quant = 16
V_max = 6.0
V_min = -6.0

# Κβάντιση των δειγμάτων
m_quantized = quantize_uniform(m_samp1, L_quant, V_min, V_max)

# Σφάλμα Κβάντισης (στο πεδίο των δειγμάτων)
quantization_error = m_quantized - m_samp1

print(f"\n--- Άσκηση 5.8-2: Κβάντιση (L={L_quant}) ---")
print(f"Δυναμική Περιοχή: [{V_min}, {V_max}]")
print(f"Βήμα Κβάντισης (Delta): {(V_max - V_min) / L_quant:.2f}")

plt.figure(figsize=(14, 6))
plt.plot(t_samp1, m_samp1, 'bo-', label='Αρχικά Δείγματα $m[n]$', markersize=4)
plt.stem(t_samp1, m_quantized, 'r', markerfmt='rs', basefmt='r-', 
         label=f'Κβαντισμένα Δείγματα $m_q[n]$ (L={L_quant})')
plt.title('Σύγκριση Αρχικών και Κβαντισμένων Δειγμάτων')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.show()

# --- (a) Ιδανική Ανακατασκευή LPF ---

# (Χρήση της συνάρτησης από την 5.8-1)
def ideal_lpf_reconstruct(signal_samples, t_samples, t_target, fs_sample):
    N_pad = len(t_target)
    M_f = fft(signal_samples, n=N_pad)
    freq_axis = fftfreq(N_pad, 1/fs_sample)
    B_lpf = fs_sample / 2
    M_f[np.abs(freq_axis) > B_lpf] = 0
    scaling_factor = N_pad / len(signal_samples)
    m_recon = np.real(ifft(M_f, n=N_pad) * scaling_factor)
    return m_recon

# Ανακατασκευή από κβαντισμένα δείγματα
m_recon_quant = ideal_lpf_reconstruct(m_quantized, t_samp1, t_cont, fs1)

# Σφάλμα: Διαφορά από το *αρχικό* συνεχές σήμα
error_quant = m_recon_quant - m_cont

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(t_cont, m_cont, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_cont, m_recon_quant, 'r--', label='Ανακατασκευή (μετά Κβάντιση)')
plt.title('(a) Ιδανική Ανακατασκευή LPF από Κβαντισμένα Δείγματα')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_cont, error_quant, 'r-')
plt.title('Διαφορά (Σφάλμα) από το Αρχικό Σήμα (Θόρυβος Κβάντισης)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Σφάλμα')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- (b) Ανακατασκευή ZOH + Equalizer ---

print("\n--- Μέρος (b): ZOH & Equalizer (σε κβαντισμένα) ---")
f_zoh_q = interp1d(t_samp1, m_quantized, kind='previous', fill_value='extrapolate')
m_recon_zoh_q = f_zoh_q(t_cont)

# Ο Equalizer είναι ο ίδιος με το 5.8-1c
N_cont = len(t_cont)
freq_axis_cont = fftshift(fftfreq(N_cont, 1/fs_cont))
epsilon = 1e-9
H_zoh = Ts1 * np.sinc(freq_axis_cont * Ts1) * np.exp(-1j * np.pi * freq_axis_cont * Ts1)
H_lpf = np.where(np.abs(freq_axis_cont) <= fs1 / 2, 1, 0)
H_eq_zoh = H_lpf / (H_zoh + epsilon)

# Εφαρμογή του ZOH Equalizer
M_recon_zoh_q_f = fftshift(fft(m_recon_zoh_q))
M_eq_zoh_f = M_recon_zoh_q_f * H_eq_zoh
m_eq_zoh = np.real(ifft(ifftshift(M_eq_zoh_f)))
error_zoh = m_eq_zoh - m_cont

# --- (c) Ανακατασκευή FOH + Equalizer ---

print("--- Μέρος (c): FOH & Equalizer (σε κβαντισμένα) ---")
# FOH = Γραμμική Παρεμβολή
f_foh_q = interp1d(t_samp1, m_quantized, kind='linear', fill_value='extrapolate')
m_recon_foh_q = f_foh_q(t_cont)

# Σχεδιασμός FOH Equalizer (H ~ sinc^2)
H_foh = Ts1 * (np.sinc(freq_axis_cont * Ts1))**2 * \
        np.exp(-1j * 2 * np.pi * freq_axis_cont * Ts1) # FOH delay = Ts
H_eq_foh = H_lpf / (H_foh + epsilon)

# Εφαρμογή του FOH Equalizer
M_recon_foh_q_f = fftshift(fft(m_recon_foh_q))
M_eq_foh_f = M_recon_foh_q_f * H_eq_foh
m_eq_foh = np.real(ifft(ifftshift(M_eq_foh_f)))
error_foh = m_eq_foh - m_cont

# --- Σύγκριση Σφαλμάτων (b) και (c) ---

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(t_cont, error_zoh, 'r-', label='Σφάλμα: ZOH + EQ')
plt.title('(c) Σύγκριση Σφαλμάτων Ανακατασκευής (από Κβαντισμένα Δείγματα)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_cont, error_foh, 'g-', label='Σφάλμα: FOH + EQ')
plt.legend()
plt.xlabel('Χρόνος (s)')
plt.ylabel('Σφάλμα')
plt.grid(True)
plt.tight_layout()
plt.show()

# Υπολογισμός Ισχύος Σφάλματος (MSE)
mse_zoh = np.mean(error_zoh**2)
mse_foh = np.mean(error_foh**2)
print(f"Mean Squared Error (ZOH + EQ): {mse_zoh:.4e}")
print(f"Mean Squared Error (FOH + EQ): {mse_foh:.4e}")
print("-> Η FOH (Γραμμική Παρεμβολή) είναι σαφώς ανώτερη.")
