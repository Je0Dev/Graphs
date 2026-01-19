import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# --- Ρύθμιση του Seaborn για "modern, clean look" ---
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['font.family'] = 'Arial' # Μπορεί να χρειαστεί αλλαγή γραμματοσειράς

# =============================================================================
# ΔΙΑΓΡΑΜΜΑ 1: Μελέτη Αυτοθέρμανσης (R = f(P))
# =============================================================================

# Δεδομένα από τον Πίνακα 1
data_air = {
    'P (mW)': [9.7, 37, 216.4, 353.4], # Χρησιμοποιώντας τη διορθωμένη τιμή
    'R (kΩ)': [2.3, 2.2, 1.53, 1.02]
}
df_air = pd.DataFrame(data_air)
df_air['Κατάσταση'] = 'Στον Αέρα'

data_water = {
    'P (mW)': [3.45, 34.03, 164, 320],
    'R (kΩ)': [2.17, 1.87, 1.62, 1.41]
}
df_water = pd.DataFrame(data_water)
df_water['Κατάσταση'] = 'Στο Νερό'

# Συνδυασμός των δεδομένων για το Seaborn
df_diag1 = pd.concat([df_air, df_water])

# Δημιουργία Γραφήματος 1
plt.figure(figsize=(10, 6))
diag1_plot = sns.lineplot(
    data=df_diag1,
    x='P (mW)',
    y='R (kΩ)',
    hue='Κατάσταση',
    style='Κατάσταση',
    markers=True,
    dashes=False,
    markersize=8,
    palette={'Στον Αέρα': 'blue', 'Στο Νερό': 'red'}
)

# Ρύθμιση λογαριθμικού άξονα Χ, όπως στο PDF
diag1_plot.set(xscale="log")

# Τίτλοι και ετικέτες
plt.title('Διάγραμμα 1: Φαινόμενο Αυτοθέρμανσης (R = f(P))', fontsize=16)
plt.xlabel('Ισχύς P (mW)', fontsize=12)
plt.ylabel('Αντίσταση R (kΩ)', fontsize=12)
plt.legend(title='Μέσο Μέτρησης')
plt.grid(True, which="both", ls="--")

# Αποθήκευση γραφήματος
plt.savefig('diagram_1_autothermansi.png', dpi=300, bbox_inches='tight')
print("Διάγραμμα 1 (diagram_1_autothermansi.png) αποθηκεύτηκε.")


# =============================================================================
# ΔΙΑΓΡΑΜΜΑ 2: Καμπύλη Βαθμονόμησης (R = f(θ))
# =============================================================================

# Δεδομένα από τον Πίνακα 2
data_calib = {
    'Θερμοκρασία (°C)': [25, 30, 37, 42, 48, 53, 59, 65, 70, 78, 83, 5.5],
    'Αντίσταση (Ω)': [2200, 2013, 1430, 1101, 1002, 810, 610, 510, 407, 320, 290, 5100]
}
df_calib = pd.DataFrame(data_calib)

# Ταξινόμηση βάσει θερμοκρασίας για σωστή γραμμή
df_calib = df_calib.sort_values(by='Θερμοκρασία (°C)')

# Δεδομένα άγνωστου σώματος
R_unknown = 533 # Αντίσταση (Ω)

# --- Παρεμβολή (Interpolation) ---
# Η σχέση είναι R=f(T), αλλά εμείς θέλουμε T=f(R)
# Χρησιμοποιούμε τις τιμές R ως x και T ως y για τη συνάρτηση παρεμβολής
# Χρησιμοποιούμε 'cubic' (κυβική) παρεμβολή για πιο ομαλή καμπύλη
f_temp_from_R = interp1d(
    df_calib['Αντίσταση (Ω)'], 
    df_calib['Θερμοκρασία (°C)'], 
    kind='linear' # Αλλάζουμε σε γραμμική, καθώς τα σημεία είναι λίγα
)

# Υπολογισμός της άγνωστης θερμοκρασίας
# Πρέπει να βρούμε τα σημεία που "αγκαλιάζουν" το 533
# (510, 65) και (610, 59)
# Χρησιμοποιούμε μια απλή γραμμική παρεμβολή μεταξύ αυτών των δύο σημείων
interp_linear = interp1d([510, 610], [65, 59])
T_unknown = interp_linear(R_unknown)

print(f"Η υπολογισμένη θερμοκρασία είναι: {T_unknown:.2f} °C")

# Δημιουργία Γραφήματος 2
plt.figure(figsize=(10, 7))

# Σημεία δεδομένων
sns.scatterplot(
    data=df_calib,
    x='Θερμοκρασία (°C)',
    y='Αντίσταση (Ω)',
    s=100,
    label='Σημεία Βαθμονόμησης'
)

# Γραμμή τάσης (που συνδέει τα σημεία)
sns.lineplot(
    data=df_calib,
    x='Θερμοκρασία (°C)',
    y='Αντίσταση (Ω)',
    color='gray',
    linestyle='--',
    label='Καμπύλη Βαθμονόμησης'
)

# Προσθήκη του άγνωστου σημείου
plt.plot(T_unknown, R_unknown, 'r*', markersize=15, label=f'Άγνωστο Σώμα ({T_unknown:.1f} °C, {R_unknown} Ω)')

# Γραμμές προβολής
plt.hlines(R_unknown, 0, T_unknown, colors='red', linestyles='dotted')
plt.vlines(T_unknown, 0, R_unknown, colors='red', linestyles='dotted')

# Τίτλοι και ετικέτες
plt.title('Διάγραμμα 2: Καμπύλη Βαθμονόμησης Θερμίστορ (R = f(θ))', fontsize=16)
plt.xlabel('Θερμοκρασία θ (°C)', fontsize=12)
plt.ylabel('Αντίσταση R (Ω)', fontsize=12)
plt.legend()
plt.xlim(0, 90)
plt.ylim(0, 5500)

# Αποθήκευση γραφήματος
plt.savefig('diagram_2_vathmonomisi.png', dpi=300, bbox_inches='tight')
print("Διάγραμμα 2 (diagram_2_vathmonomisi.png) αποθηκεύτηκε.")

# Εμφάνιση γραφημάτων
plt.show()
