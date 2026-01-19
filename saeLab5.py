import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm

# Ρύθμιση για εμφάνιση ελληνικών χαρακτήρων
plt.rcParams['font.family'] = 'DejaVu Sans'  # Ή άλλη γραμματοσειρά που υποστηρίζει ελληνικούς χαρακτήρες
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 8),
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

# Καθαρό επαγγελματικό στυλ
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# ==============================
# ΔΕΔΟΜΕΝΑ ΑΠΟ ΤΑ ΠΕΙΡΑΜΑΤΑ
# ==============================

# Πίνακας 1: Διάταξη με έναν αισθητήρα (Βήμα 3)
strain_single = np.array([0, 350, 700, 1050, 1400, 1750, 2100, 2450, 2800, 3250, 3600])  # σε με
voltage_single = np.array([0, 1.01, 1.81, 2.61, 3.40, 4.20, 5.04, 5.89, 6.73, 7.59, 8.42])  # σε V

# Πίνακας 2: Διάταξη με δύο αισθητήρες (Βήματα 5-6)
strain_dual = np.array([0, 350, 700, 1050, 1400, 1750, 2100, 2450, 2800, 3250, 3600])  # σε με
voltage_loading = np.array([0, 1.75, 3.31, 4.90, 6.43, 8.00, 9.70, 11.40, 13.13, 14.03, 14.09])  # Φόρτωση (Βήμα 5)
voltage_unloading = np.array([0, 1.62, 3.12, 4.62, 6.14, 7.74, 9.34, 11.03, 12.60, 13.96, 14.08])  # Εκφόρτωση (Βήμα 6)

# ==============================
# ΥΠΟΛΟΓΙΣΜΟΙ ΓΙΑ ΤΗ ΜΟΝΗ ΑΙΣΘΗΤΗΡΑ
# ==============================
FSI_single = strain_single[-1]  # Πλήρης Κλίμακα Εισόδου = μέγιστη παραμόρφωση = 3600 με
FSO_single = voltage_single[-1]  # Πλήρης Κλίμακα Εξόδου = έξοδος στο FSI = 8.42 V
sensitivity_single = FSO_single / FSI_single  # Ευαισθησία σε V/με

# Θεωρητική καμπύλη για τη διάταξη με έναν αισθητήρα
G_f = 2.0  # Συντελεστής αισθητήρα
V_ex = 2.0  # Τάση διέγερσης (V)
R = 120.0   # Αντίσταση βάσης (Ω)
Av = 2000   # Κέρδος ενισχυτή

def theoretical_voltage_single(strain_με):
    """Υπολογισμός θεωρητικής τάσης εξόδου για διάταξη με έναν αισθητήρα"""
    strain = strain_με * 1e-6  # Μετατροπή με σε αδιάστατη παραμόρφωση
    delta_R = G_f * R * strain      # Αλλαγή αντίστασης
    # Ακριβής φόρμουλα γέφυρας Wheatstone για τετραμερή γέφυρα
    V_b = V_ex * (delta_R / (4 * R + 2 * delta_R))  # Τάση εξόδου γέφυρας
    V_out = Av * V_b           # Ενισχυμένη έξοδος
    return V_out

strain_fine = np.linspace(0, 3600, 100)  # Λεπτές τιμές παραμόρφωσης για ομαλή καμπύλη
voltage_theoretical_single = theoretical_voltage_single(strain_fine)

# ==============================
# ΓΡΑΦΙΚΗ 1: ΜΟΝΟΣ ΑΙΣΘΗΤΗΡΑΣ ΜΕ ΟΛΕΣ ΤΙΣ ΤΙΜΕΣ ΣΗΜΕΙΩΝ
# ==============================
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Θεωρητική καμπύλη (παρασκήνιο)
sns.lineplot(x=strain_fine, y=voltage_theoretical_single, color='#d62728', linewidth=2.5, 
             label='Θεωρητική Καμπύλη', zorder=2)

# Πειραματικά σημεία με σαφείς δείκτες
scatter = ax.scatter(strain_single, voltage_single, s=80, color='#1f77b4', edgecolor='black', 
                     linewidth=1, zorder=5, label='Πειραματικά Δεδομένα')

# Επισήμανση σημείων FSI και FSO με ειδικούς δείκτες
ax.scatter([FSI_single], [FSO_single], s=150, color='#2ca02c', edgecolor='black', 
           linewidth=1.5, zorder=6, marker='*', label=f'Σημείο FSO ({FSI_single}, {FSO_single:.2f})')

# Γραμμή ευαισθησίας από την αρχή στο σημείο FSO
ax.plot([0, FSI_single], [0, FSO_single], color='#9467bd', linestyle='--', linewidth=2, 
        label=f'Ευαισθησία = {sensitivity_single:.4f} V/μ\u03B5')

# Οριζόντιες και κάθετες γραμμές για FSI και FSO
ax.axvline(x=FSI_single, color='#ff7f0e', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
ax.axhline(y=FSO_single, color='#ff7f0e', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)

# Προσθήκη ετικετών για ΚΑΘΕ σημείο με τις συντεταγμένες (x,y)
for i, (x, y) in enumerate(zip(strain_single, voltage_single)):
    # Προσαρμογή θέσης ετικέτας για αποφυγή επικάλυψης
    offset_x = 0
    offset_y = 8 if i % 2 == 0 else -15
    
    ax.annotate(f'({x}, {y:.2f})', 
               (x, y), 
               xytext=(offset_x, offset_y), 
               textcoords='offset points',
               ha='center', 
               va='bottom' if i % 2 == 0 else 'top',
               fontsize=9,
               alpha=0.9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
               fontweight='bold')

# Ονομασίες αξόνων και τίτλος
ax.set_xlabel('Μηχανική Παραμόρφωση (\u03BC\u03B5)', fontsize=14, fontweight='bold')
ax.set_ylabel('Τάση Εξόδου (V)', fontsize=14, fontweight='bold')
ax.set_title('Διάγραμμα 1: Διάταξη με Έναν Αισθητήρα - Πειραματικά Δεδομένα και Θεωρητικό Μοντέλο', 
             fontsize=16, fontweight='bold', pad=15)

# Όρια αξόνων με περιθώριο
ax.set_xlim(-200, 3800)
ax.set_ylim(-0.5, 9.5)

# Ρύθμιση ticks
ax.set_xticks(np.arange(0, 3800, 500))
ax.xaxis.set_major_locator(MaxNLocator(9))
ax.yaxis.set_major_locator(MaxNLocator(7))

# Πλέγμα για καλύτερη αναγνωσιμότητα
ax.grid(True, linestyle='--', alpha=0.7)

# Πλαίσιο υπολογισμών στην κάτω δεξιά γωνία
textstr = (f'Πλήρης Κλίμακα Εισόδου (FSI): {FSI_single} \u03BC\u03B5\n'
           f'Πλήρης Κλίμακα Εξόδου (FSO): {FSO_single:.2f} V\n'
           f'Ευαισθησία: {sensitivity_single:.4f} V/\u03BC\u03B5\n'
           f'Συντελεστής Αισθητήρα (G): {G_f}\n'
           f'Ενίσχυση: {Av}x')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, 
             edgecolor='darkgoldenrod', linewidth=1.5)
ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', 
        bbox=props, fontweight='bold')

# Τοποθέτηση legend σε κατάλληλη θέση
ax.legend(loc='upper left', frameon=True, shadow=True, framealpha=0.95,
         title='Υπολογισμοί',
         title_fontproperties={'weight': 'bold', 'size': 13})

plt.tight_layout()
plt.savefig('diagram1_single_sensor_greek.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# ΓΡΑΦΙΚΗ 2: ΔΙΠΛΟΣ ΑΙΣΘΗΤΗΡΑΣ ΜΕ ΟΛΑ ΤΑ ΣΗΜΕΙΑ ΚΑΙ ΥΣΤΕΡΗΣΗ
# ==============================
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Θεωρητική καμπύλη
def theoretical_voltage_dual(strain_με):
    """Υπολογισμός θεωρητικής τάσης εξόδου για διάταξη με δύο αισθητήρες (ημιγέφυρα)"""
    strain = strain_με * 1e-6  # Μετατροπή με σε αδιάστατη παραμόρφωση
    delta_R = G_f * R * strain      # Μέγεθος αλλαγής αντίστασης
    # Ακριβής φόρμουλα γέφυρας Wheatstone για διαφορική ημιγέφυρα
    V_b = V_ex * (2 * delta_R * R) / (4 * R**2 - delta_R**2)
    V_out = Av * V_b
    return V_out

sns.lineplot(x=strain_fine, y=theoretical_voltage_dual(strain_fine), color='#d62728', 
             linestyle='--', linewidth=2.5, label='Θεωρητική Καμπύλη', zorder=2)

# Πειραματικά δεδομένα φόρτωσης/εκφόρτωσης
ax.plot(strain_dual, voltage_loading, 'bo-', markersize=8, linewidth=2.5, 
        label='Φόρτωση (Βήμα 5)', zorder=5)
ax.plot(strain_dual, voltage_unloading, 'gs-', markersize=8, linewidth=2.5, 
        label='Εκφόρτωση (Βήμα 6)', zorder=5)

# Επισήμανση FSI και FSO
FSI_dual = strain_dual[-1]
FSO_dual = voltage_loading[-1]
sensitivity_dual = FSO_dual / FSI_dual

ax.scatter([FSI_dual], [FSO_dual], s=150, color='#2ca02c', edgecolor='black', 
           linewidth=1.5, zorder=6, marker='*', label=f'Σημείο FSO ({FSI_dual}, {FSO_dual:.2f})')

# Γραμμή ευαισθησίας
ax.plot([0, FSI_dual], [0, FSO_dual], color='#9467bd', linestyle='--', linewidth=2, 
        label=f'Ευαισθησία = {sensitivity_dual:.4f} V/μ\u03B5')

# Υπολογισμός υστέρησης
hysteresis_diff = np.abs(voltage_loading - voltage_unloading)
max_hysteresis = np.max(hysteresis_diff)
max_hysteresis_idx = np.argmax(hysteresis_diff)
max_hysteresis_strain = strain_dual[max_hysteresis_idx]
hysteresis_percent = (max_hysteresis / FSO_dual) * 100

# Επισήμανση μέγιστης υστέρησης
ax.plot([max_hysteresis_strain, max_hysteresis_strain], 
        [voltage_loading[max_hysteresis_idx], voltage_unloading[max_hysteresis_idx]],
        color='#8c564b', linewidth=4, alpha=0.8, label='Μέγιστη Υστέρηση')

# Ετικέτες για ΚΑΘΕ σημείο φόρτωσης
for i, (x, y) in enumerate(zip(strain_dual, voltage_loading)):
    offset_y = 8 if i % 2 == 0 else -15
    ax.annotate(f'Φ({x}, {y:.2f})', 
               (x, y), 
               xytext=(0, offset_y), 
               textcoords='offset points',
               ha='center', 
               va='bottom' if i % 2 == 0 else 'top',
               fontsize=8,
               alpha=0.9,
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", alpha=0.6),
               fontweight='bold')

# Ετικέτες για ΚΑΘΕ σημείο εκφόρτωσης
for i, (x, y) in enumerate(zip(strain_dual, voltage_unloading)):
    offset_y = 8 if i % 2 == 1 else -15
    ax.annotate(f'Ε({x}, {y:.2f})', 
               (x, y), 
               xytext=(20, offset_y), 
               textcoords='offset points',
               ha='center', 
               va='bottom' if i % 2 == 1 else 'top',
               fontsize=8,
               alpha=0.9,
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.6),
               fontweight='bold')

# Ονομασίες αξόνων και τίτλος
ax.set_xlabel('Μηχανική Παραμόρφωση (\u03BC\u03B5)', fontsize=14, fontweight='bold')
ax.set_ylabel('Τάση Εξόδου (V)', fontsize=14, fontweight='bold')
ax.set_title('Διάγραμμα 2: Διάταξη με Δύο Αισθητήρες - Φόρτωση/Εκφόρτωση και Υστέρηση', 
             fontsize=16, fontweight='bold', pad=15)

# Όρια αξόνων με περιθώριο
ax.set_xlim(-200, 3800)
ax.set_ylim(-0.5, 15.5)

# Ρύθμιση ticks
ax.set_xticks(np.arange(0, 3800, 500))
ax.xaxis.set_major_locator(MaxNLocator(9))
ax.yaxis.set_major_locator(MaxNLocator(8))

# Πλέγμα
ax.grid(True, linestyle='--', alpha=0.7)

# Πλαίσιο υπολογισμών
textstr = (f'Πλήρης Κλίμακα Εισόδου (FSI): {FSI_dual} \u03BC\u03B5\n'
           f'Πλήρης Κλίμακα Εξόδου (FSO): {FSO_dual:.2f} V\n'
           f'Ευαισθησία: {sensitivity_dual:.4f} V/\u03BC\u03B5\n'
           f'Μέγιστη Υστέρηση: {max_hysteresis:.3f} V\n'
           f'({hysteresis_percent:.2f}% του FSO)')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, 
             edgecolor='darkgoldenrod', linewidth=1.5)
ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', 
        bbox=props, fontweight='bold')

# Τοποθέτηση legend
ax.legend(loc='upper left', frameon=True, shadow=True, framealpha=0.95,
         title='Υπολογισμοί',
         title_fontproperties={'weight': 'bold', 'size': 13})

plt.tight_layout()
plt.savefig('diagram2_dual_sensor_greek.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# ΠΕΡΙΛΗΨΗ ΥΠΟΛΟΓΙΣΜΩΝ
# ==============================
print("\n" + "="*80)
print("ΥΠΟΛΟΓΙΣΜΟΙ ΓΙΑ ΤΗ ΔΙΑΤΑΞΗ ΜΕ ΕΝΑΝ ΑΙΣΘΗΤΗΡΑ".center(80))
print("="*80)
print(f"Πλήρης Κλίμακα Εισόδου (FSI):  {FSI_single} με")
print(f"Πλήρης Κλίμακα Εξόδου (FSO): {FSO_single:.4f} V")
print(f"Ευαισθησία:                  {sensitivity_single:.6f} V/με")
print(f"Θεωρητική Ευαισθησία:        0.002000 V/με")

print("\n" + "="*80)
print("ΥΠΟΛΟΓΙΣΜΟΙ ΓΙΑ ΤΗ ΔΙΑΤΑΞΗ ΜΕ ΔΥΟ ΑΙΣΘΗΤΗΡΕΣ".center(80))
print("="*80)
print(f"Πλήρης Κλίμακα Εισόδου (FSI):  {FSI_dual} με")
print(f"Πλήρης Κλίμακα Εξόδου (FSO): {FSO_dual:.4f} V")
print(f"Ευαισθησία:                  {sensitivity_dual:.6f} V/με")
print(f"Θεωρητική Ευαισθησία:        0.004000 V/με")
print(f"Μέγιστο Σφάλμα Υστέρησης:    {max_hysteresis:.4f} V ({hysteresis_percent:.2f}% του FSO)")

print("\n" + "="*80)
print("ΣΥΓΚΡΙΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ".center(80))
print("="*80)
sensitivity_ratio = sensitivity_dual / sensitivity_single
print(f"Η ευαισθησία της διάταξης με δύο αισθητήρες είναι {sensitivity_ratio:.2f} φορές μεγαλύτερη")
print(f"Θεωρητική βελτίωση που αναμενόταν: 2.00 φορές")
print("="*80)
