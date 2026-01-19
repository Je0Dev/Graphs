import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. ΕΙΣΑΓΩΓΗ ΚΑΙ ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ==========================================
data = {
    'Thesi_mm': [
        25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 
        5, 4, 3, 2, 1
    ],
    'Tasi_Vpp': [
        0.90, 0.72, 0.72, 0.60, 0.52, 0.48, 0.40, 0.30, 0.24, 0.16,
        0.10, 0.04, 0.10, 0.16, 0.20, 0.28, 0.50, 0.52, 0.68, 0.72,
        0.90, 0.86, 0.94, 0.68, 0.70
    ],
    'Fasi_deg': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 180, 180, 180, 180, 180, 180, 180, 180,
        180, 180, 180, 180, 180
    ]
}

df = pd.DataFrame(data)
df['Tasi_Final'] = df.apply(lambda row: row['Tasi_Vpp'] if row['Fasi_deg'] == 0 else -row['Tasi_Vpp'], axis=1)

# Ορισμός Ορίων Γραμμικής Περιοχής
LIMIT_MIN_MM = 6
LIMIT_MAX_MM = 22

# Κατηγοριοποίηση
df['Region'] = df['Thesi_mm'].apply(lambda x: 'Γραμμική Περιοχή' if LIMIT_MIN_MM <= x <= LIMIT_MAX_MM else 'Περιοχή Κόρου')
linear_df = df[df['Region'] == 'Γραμμική Περιοχή'].copy()

# ==========================================
# 2. ΥΠΟΛΟΓΙΣΜΟΙ (Sensitivity, FSO, FSI, Error)
# ==========================================
coefficients = np.polyfit(linear_df['Thesi_mm'], linear_df['Tasi_Final'], 1)
slope = coefficients[0]      
intercept = coefficients[1]
polynomial = np.poly1d(coefficients)

FSI = LIMIT_MAX_MM - LIMIT_MIN_MM 
v_start = polynomial(LIMIT_MIN_MM) 
v_end = polynomial(LIMIT_MAX_MM)   
FSO = abs(v_start - v_end)

linear_df['Predicted'] = polynomial(linear_df['Thesi_mm'])
linear_df['Error'] = abs(linear_df['Tasi_Final'] - linear_df['Predicted'])

max_err_idx = linear_df['Error'].idxmax()
max_err_val = linear_df.loc[max_err_idx, 'Error'] 
percent_error = (max_err_val / FSO) * 100         

err_x = linear_df.loc[max_err_idx, 'Thesi_mm']
err_y_real = linear_df.loc[max_err_idx, 'Tasi_Final']
err_y_pred = linear_df.loc[max_err_idx, 'Predicted']

# ==========================================
# 3. ΣΧΕΔΙΑΣΗ ΓΡΑΦΗΜΑΤΟΣ
# ==========================================
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(16, 12))

# Καμπύλη (Μαύρη)
x_smooth = np.linspace(df['Thesi_mm'].min(), df['Thesi_mm'].max(), 300)
spl = make_interp_spline(df['Thesi_mm'][::-1], df['Tasi_Final'][::-1], k=3)
y_smooth = spl(x_smooth)
ax.plot(x_smooth, y_smooth, color='black', linewidth=2.5, zorder=1)

# Ευθεία (Πράσινη)
x_line = np.linspace(1, 25, 100)
y_line = polynomial(x_line)
ax.plot(x_line, y_line, color='green', linestyle='--', alpha=0.6, zorder=2)

# Σημεία
sns.scatterplot(
    data=df, x='Thesi_mm', y='Tasi_Final', hue='Region', 
    palette={'Γραμμική Περιοχή': 'blue', 'Περιοχή Κόρου': 'red'},
    s=120, edgecolor='black', zorder=10, ax=ax, legend=False 
)

# Ετικέτες Τιμών
for i, row in df.iterrows():
    ax.text(
        row['Thesi_mm'], row['Tasi_Final'] + 0.04, 
        f"({int(row['Thesi_mm'])}, {row['Tasi_Final']:.2f})",
        color='black', fontsize=8, ha='center', fontweight='bold', zorder=15
    )

# ==========================================
# 4. ΟΠΤΙΚΟΠΟΙΗΣΗ ΜΕΤΡΗΣΕΩΝ (ΒΕΛΗ)
# ==========================================
# Βέλος Σφάλματος
ax.annotate('', xy=(err_x, err_y_real), xytext=(err_x, err_y_pred),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(err_x + 0.5, (err_y_real + err_y_pred)/2, 'ΜΣΜΓ', color='red', fontweight='bold', va='center')

# Βέλος FSI
ax.axvline(LIMIT_MIN_MM, color='pink', linestyle=':', alpha=0.5)
ax.axvline(LIMIT_MAX_MM, color='pink', linestyle=':', alpha=0.5)
y_fsi_pos = -1.1 
ax.annotate('', xy=(LIMIT_MIN_MM, y_fsi_pos), xytext=(LIMIT_MAX_MM, y_fsi_pos),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text((LIMIT_MIN_MM + LIMIT_MAX_MM)/2, y_fsi_pos + 0.05, f'FSI = {FSI}mm', 
        color='orange', fontweight='bold', ha='center')

# Βέλος FSO
x_fso_pos = 23.5 
ax.plot([LIMIT_MIN_MM, x_fso_pos], [v_start, v_start], color='pink', linestyle=':', alpha=0.4)
ax.plot([LIMIT_MAX_MM, x_fso_pos], [v_end, v_end], color='pink', linestyle=':', alpha=0.4)
ax.annotate('', xy=(x_fso_pos, v_start), xytext=(x_fso_pos, v_end),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text(x_fso_pos + 0.2, (v_start + v_end)/2, f'FSO\n{FSO:.2f}V', 
        color='orange', fontweight='bold', va='center')

# ==========================================
# 5. ΠΛΑΙΣΙΑ ΠΛΗΡΟΦΟΡΙΩΝ (Info Boxes)
# ==========================================

# --- Α. ΚΟΥΤΙ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (Πάνω Αριστερά) ---
info_text = (
    f"$\\bf{{ΑΠΟΤΕΛΕΣΜΑΤΑ & ΑΝΑΛΥΣΗ}}$\n"
    f"-----------------------------\n"
    f"• Ευαισθησία (S): {abs(slope):.4f} V/mm\n"
    f"• Εύρος Εισόδου (FSI): {FSI} mm\n"
    f"• Εύρος Εξόδου (FSO): {FSO:.4f} V\n"
    f"• Μέγιστο Σφάλμα (ΜΣΜΓ): {max_err_val:.4f} V\n"
    f"• Σχετικό Σφάλμα (%): {percent_error:.2f}%\n"
    f"-----------------------------\n"
)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
# Τοποθέτηση ψηλά αριστερά (y=0.98)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# --- Β. ΥΠΟΜΝΗΜΑ / LEGEND (Ακριβώς από κάτω) ---
# Δημιουργούμε χειροκίνητα τα στοιχεία του Legend για να έχουμε τον απόλυτο έλεγχο
legend_elements = [
    Line2D([0], [0], color='black', lw=2, label='Πειραματική Καμπύλη'),
    Line2D([0], [0], color='green', lw=2, linestyle='--', label='Βέλτιστη Ευθεία'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, markeredgecolor='k', label='Γραμμική Περιοχή'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='k', label='Περιοχή Κόρου'),
]

# Τοποθετούμε το Legend με bbox_to_anchor. 
# Το (0.02, 0.72) σημαίνει: 2% από αριστερά, και στο 72% του ύψους (δηλαδή κάτω από το πάνω κουτί)
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.72), 
          frameon=True, shadow=True, title="Υπόμνημα Συμβόλων")

# ==========================================
# 6. ΤΕΛΙΚΗ ΜΟΡΦΟΠΟΙΗΣΗ
# ==========================================
ax.axhline(0, color='black', linewidth=1)
ax.set_title('Πλήρης Χαρακτηριστική LVDT & Ανάλυση Σφαλμάτων', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Μετατόπιση Πυρήνα (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Τάση Εξόδου (V)', fontsize=14, fontweight='bold')

plt.xticks(np.arange(1, 26, 1))
plt.yticks(np.arange(-1.2, 1.4, 0.2))

plt.tight_layout()
plt.show()

