import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ορισμός Σήματος και Συναρτήσης DM ---

f1 = 400; f2 = 800; f3 = 1200;
def m(t):
    return 2 * np.cos(2 * np.pi * f1 * t) + \
           np.cos(2 * np.pi * f2 * t) - \
           3 * np.sin(2 * np.pi * f3 * t)

def simulate_dm(signal_samples, E_step):
    """
    Προσομοιώνει τον αλγόριθμο Delta Modulation (DM).
    """
    N = len(signal_samples)
    m_quantized = np.zeros(N) # Η έξοδος "σκάλα"
    
    # Αρχικοποίηση (μπορεί να είναι 0 ή η πρώτη τιμή)
    m_quantized[0] = signal_samples[0] 
    
    for n in range(1, N):
        # 1. Βρίσκουμε τη διαφορά
        error = signal_samples[n] - m_quantized[n-1]
        
        # 2. Κβάντιση 1-bit
        step = E_step * np.sign(error)
        
        # 3. Συσσώρευση
        m_quantized[n] = m_quantized[n-1] + step
        
    return m_quantized

# --- 2. Δημιουργία Δειγμάτων για DM ---
# Η DM απαιτεί τα δικά της δείγματα, δεν χρησιμοποιεί τα m_samp1
T_duration = 0.02

# (a) & (b) Παράμετροι
fs_a = 9600
Ts_a = 1/fs_a
t_dm_a = np.arange(0, T_duration, Ts_a)
m_dm_a = m(t_dm_a) # Αρχικό σήμα δειγματοληπτημένο στα 9600 Hz

# --- (a) fs = 9600 Hz, E = 0.2 (Slope Overload) ---

print("\n--- Άσκηση 5.8-3: Delta Modulation ---")
E_a = 0.2
m_q_a = simulate_dm(m_dm_a, E_a)
max_slope_signal = 32671
max_slope_dm_a = E_a * fs_a
print(f"(a) fs={fs_a} Hz, E={E_a}")
print(f"Max Signal Slope: ~{max_slope_signal:.0f} V/s")
print(f"Max DM Slope: {max_slope_dm_a:.0f} V/s")
print("-> Slope Overload: YES")

plt.figure(figsize=(14, 6))
plt.plot(t_dm_a, m_dm_a, 'b-', label='Αρχικό $m(t)$ @ 9600 Hz')
plt.plot(t_dm_a, m_q_a, 'r-', label=f'DM Έξοδος $m_q[n]$ (E={E_a})')
plt.title('(a) DM με Σοβαρή Υπερφόρτωση Κλίσης (Slope Overload)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.show()

# --- (b) fs = 9600 Hz, Μεταβολή E ---

# Αύξηση E (Μείωση Overload, Αύξηση Granular Noise)
E_b1 = 1.0 
m_q_b1 = simulate_dm(m_dm_a, E_b1)
print(f"(b) Max DM Slope (E={E_b1}): {E_b1 * fs_a:.0f} V/s")

# Μείωση E (Αύξηση Overload, Μείωση Granular Noise)
E_b2 = 0.05
m_q_b2 = simulate_dm(m_dm_a, E_b2)
print(f"(b) Max DM Slope (E={E_b2}): {E_b2 * fs_a:.0f} V/s")

plt.figure(figsize=(14, 12))
plt.subplot(2, 1, 1)
plt.plot(t_dm_a, m_dm_a, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_dm_a, m_q_b1, 'g-', label=f'DM (E={E_b1}) - Λιγότερο Overload')
plt.title(f'(b) Αύξηση E: Μείωση Overload, Αύξηση Granular Noise')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_dm_a, m_dm_a, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_dm_a, m_q_b2, 'm-', label=f'DM (E={E_b2}) - Περισσότερο Overload')
plt.title(f'(b) Μείωση E: Αύξηση Overload, Μείωση Granular Noise')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- (c) Βελτίωση Παραμέτρων DM ---

# Στρατηγική 1 (Κακή): fs = 9600 Hz, E = 3.5
E_c1 = 3.5
m_q_c1 = simulate_dm(m_dm_a, E_c1)
print(f"(c.1) Max DM Slope (E={E_c1}): {E_c1 * fs_a:.0f} V/s (OK)")

# Στρατηγική 2 (Καλή): fs = 200 kHz, E = 0.2
fs_c2 = 200000
Ts_c2 = 1/fs_c2
t_dm_c2 = np.arange(0, T_duration, Ts_c2)
m_dm_c2 = m(t_dm_c2) # Νέα δειγματοληψία

E_c2 = 0.2
m_q_c2 = simulate_dm(m_dm_c2, E_c2)
print(f"(c.2) Max DM Slope (fs={fs_c2}, E={E_c2}): {E_c2 * fs_c2:.0f} V/s (OK)")


plt.figure(figsize=(14, 12))
plt.subplot(2, 1, 1)
plt.plot(t_dm_a, m_dm_a, 'b-', label='Αρχικό $m(t)$')
plt.plot(t_dm_a, m_q_c1, 'r-', label=f'DM (fs=9600, E={E_c1})')
plt.title('(c) Στρατηγική 1: Τεράστιος Κοκκώδης Θόρυβος')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_dm_c2, m_dm_c2, 'b-', label='Αρχικό $m(t)$ @ 200kHz')
plt.plot(t_dm_c2, m_q_c2, 'g-', label=f'DM (fs=200k, E={E_c2})')
plt.title('(c) Στρατηγική 2: Καλή Παρακολούθηση (Oversampling)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Ανάκτηση Σήματος (DM Recovery) ---
# Η ανάκτηση γίνεται με LPF. Ας χρησιμοποιήσουμε την 'καλή' DM
from scipy.signal import butter, lfilter

# Σχεδιασμός LPF (π.χ., 2000 Hz)
B_lpf = 2000
nyquist = fs_c2 / 2
b, a = butter(5, B_lpf / nyquist, btype='low')

# Φιλτράρισμα της "σκάλας" DM
m_recovered_dm = lfilter(b, a, m_q_c2)

plt.figure(figsize=(14, 6))
plt.plot(t_dm_c2, m_dm_c2, 'b-', label='Αρχικό $m(t)$ @ 200kHz')
plt.plot(t_dm_c2, m_recovered_dm, 'g--', label='Ανακτημένο Σήμα (μετά LPF)')
plt.title('(c) Ανάκτηση Σήματος DM με Χαμηλοπερατό Φίλτρο (LPF)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.show()
