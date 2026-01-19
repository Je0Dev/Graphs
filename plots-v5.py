import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

# --- ΝΕΑ ΒΙΒΛΙΟΘΗΚΗ ---
# (Πρέπει να κάνεις 'pip install adjustText')
from adjustText import adjust_text

# --- 0. Στοιχεία Φοιτητή ---
USER_ID = "$id$"
sns.set_theme(style="whitegrid", palette="deep")

# --- 1. Δεδομένα από τους Πίνακες του .tex (Lab 3) ---

# --- 1A. Δεδομένα Ενότητας Α (Προενισχυτής) ---
vi_A = np.array([
    1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
    -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5
])
vo_pin3_A = np.array([
    13.61, 13.61, 13.61, 13.35, 12.34, 11.39, 10.34, 9.25, 8.27, 7.18, 6.16, 5.1, 4.11, 3.04, 2.01, 0.99,
    0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96
])
vo_pin4_A = np.array([
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.985,
    2.02, 3.07, 4.09, 5.1, 6.14, 7.24, 8.21, 9.26, 10.35, 11.34, 12.37, 13.44, 13.61, 13.61, 13.61
])
vo_diff_A = np.array([
    -12.67, -12.67, -12.67, -12.44, -11.4, -10.45, -9.39, -8.3, -7.33, -6.24, -5.21, -4.16, -3.16, -2.09, -1.05, -0.01,
    1.07, 2.11, 3.13, 4.14, 5.18, 6.28, 7.26, 8.30, 9.39, 10.38, 11.41, 12.48, 12.65, 12.65, 12.65
])

# --- 1B. Δεδομένα Ενότητας Β (Σήμα Σφάλματος) ---
angle_in_B2 = np.array([0, 30, 60, 90, 120, 150, 160, -30, -60, -90, -120, -150, -160])
vo_B2 = np.array([0.00, -2.62, -5.23, -7.73, -10.24, -12.20, -12.20, 2.42, 4.89, 7.38, 9.98, 12.45, 12.45])

angle_in_B3 = np.array([0, 30, 60, 90, 100, 120, 150, -30, -60, -90, -120, -140, -150, -160])
vo_B3 = np.array([-5.15, -7.73, -10.39, -12.20, -12.20, -12.20, -12.20, -2.70, -0.26, 2.25, 4.83, 6.60, 7.32, 7.32])

# --- 1C. Δεδομένα Ενότητας Γ (Νεκρή Ζώνη) ---
gain_C5 = np.array([0.5, 1, 1.5, 2])
deadband_cw_C5 = np.array([5, 3, 1, 0.2])
deadband_ccw_C5 = np.array([5, 3, 1, 0.2])
deadband_total_C5 = deadband_cw_C5 + deadband_ccw_C5


# --- 2. Βοηθητικές Συναρτήσεις Γραφικών (Ενημερωμένες) ---

infobox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

def plot_characteristic(x, y, x_label, y_label, title, data_label, save_path, 
                        show_regression=True, linear_range=None, 
                        show_annotations=True):
    """
    Δημιουργεί ένα γράφημα, συνδέοντας τα σημεία.
    Χρησιμοποιεί adjustText για να "απλώσει" τις ετικέτες.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    x_min, x_max = np.min(x), np.max(x)
    text_x_pos = 0.05 if x_min >= 0 else 0.65 

    ax.plot(x, y, 'o-', label=data_label, zorder=5) 
    
    # --- ΑΛΛΑΓΗ: Χρήση adjustText για καθαρές ετικέτες ---
    if show_annotations:
        texts = []
        is_float_plot = "Τάση" in y_label or "Ρεύμα" in y_label or "Τάση" in x_label or "Ρεύμα" in x_label
        for i in range(len(x)):
            if is_float_plot:
                label_text = f'({x[i]:.2f}, {y[i]:.2f})'
            else: 
                label_text = f'({x[i]:.0f}, {y[i]:.0f})'
            
            # Απλά προσθέτουμε τα αντικείμενα κειμένου σε μια λίστα
            texts.append(ax.text(x[i], y[i], label_text, 
                                 ha='center', fontsize=7.5, alpha=0.7))
        
        # Η συνάρτηση adjust_text "απλώνει" τις ετικέτες αυτόματα
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
    # --- ΤΕΛΟΣ ΑΛΛΑΓΗΣ ---
    
    if show_regression:
        x_reg, y_reg = x, y
        if linear_range:
            indices = np.where((x >= linear_range[0]) & (x <= linear_range[1]))
            x_reg, y_reg = x[indices], y[indices]
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_reg, y_reg)
        x_fit_line = np.linspace(x_min, x_max, 100)
        y_fit_line = slope * x_fit_line + intercept
        ax.plot(x_fit_line, y_fit_line, color='red', linestyle='--', label='Γραμμή Παλινδρόμησης', zorder=10, linewidth=2)
        
        regression_text = (
            f'Γραμμή Παλινδρόμησης:\n'
            f'$y = {slope:.2f}x + {intercept:.2f}$\n'
            f'$R^2 = {r_value**2:.4f}$'
        )
        ax.text(text_x_pos, 0.05, regression_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=infobox_props, zorder=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    full_title = f"{title}\n{USER_ID}"
    ax.set_title(full_title, fontweight='bold', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200) 
    plt.close(fig)
    print(f"Δημιουργήθηκε το γράφημα: {save_path}")

def plot_comparison(x1, y1, label1, x2, y2, label2, x_label, y_label, title, save_path,
                    linear_range1=None, linear_range2=None, 
                    show_annotations=True):
    """
    Δημιουργεί ένα γράφημα σύγκρισης.
    Χρησιμοποιεί adjustText για να "απλώσει" τις ετικέτες.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    is_float_plot = "Τάση" in y_label or "Ρεύμα" in y_label or "Τάση" in x_label or "Ρεύμα" in x_label

    # --- Δεδομένα 1 ---
    plot1_color = ax.plot(x1, y1, 'o-', label=label1, alpha=0.8, zorder=5)[0].get_color()
    
    x1_reg, y1_reg = x1, y1
    if linear_range1:
        indices = np.where((x1 >= linear_range1[0]) & (x1 <= linear_range1[1]))
        x1_reg, y1_reg = x1[indices], y1[indices]
            
    slope1, intercept1, r1, _, _ = stats.linregress(x1_reg, y1_reg)
    x1_fit_line = np.linspace(np.min(x1), np.max(x1), 100)
    y_fit1 = slope1 * x1_fit_line + intercept1
    ax.plot(x1_fit_line, y_fit1, linestyle='--', label=f'Παλινδρόμηση ({label1})', zorder=10, linewidth=2, color=plot1_color)

    # --- Δεδομένα 2 ---
    plot2_color = ax.plot(x2, y2, 's-', label=label2, alpha=0.8, zorder=5)[0].get_color()
    
    x2_reg, y2_reg = x2, y2
    if linear_range2:
        indices = np.where((x2 >= linear_range2[0]) & (x2 <= linear_range2[1]))
        x2_reg, y2_reg = x2[indices], y2[indices]
            
    slope2, intercept2, r2, _, _ = stats.linregress(x2_reg, y2_reg)
    x2_fit_line = np.linspace(np.min(x2), np.max(x2), 100)
    y_fit2 = slope2 * x2_fit_line + intercept2
    ax.plot(x2_fit_line, y_fit2, linestyle=':', label=f'Παλινδρόμηση ({label2})', zorder=10, linewidth=2, color=plot2_color)
    

    # --- ΑΛΛΑΓΗ: Χρήση adjustText για ΟΛΕΣ τις ετικέτες ---
    if show_annotations:
        texts = []
        # Βρόχος 1: Ετικέτες για Δεδομένα 1
        for i in range(len(x1)):
            if is_float_plot: label_text = f'({x1[i]:.2f}, {y1[i]:.2f})'
            else: label_text = f'({x1[i]:.0f}, {y1[i]:.0f})'
            texts.append(ax.text(x1[i], y1[i], label_text, 
                                 ha='center', fontsize=7.5, alpha=0.7, color=plot1_color))
        
        # Βρόχος 2: Ετικέτες για Δεδομένα 2
        for i in range(len(x2)):
            if is_float_plot: label_text = f'({x2[i]:.2f}, {y2[i]:.2f})'
            else: label_text = f'({x2[i]:.0f}, {y2[i]:.0f})'
            texts.append(ax.text(x2[i], y2[i], label_text, 
                                 ha='center', fontsize=7.5, alpha=0.7, color=plot2_color))

        # "Απλώνουμε" όλες τις ετικέτες μαζί
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
    # --- ΤΕΛΟΣ ΑΛΛΑΓΗΣ ---
    
    # Κείμενο για το infobox
    regression_text = (
        f'{label1}:\n'
        f'$y = {slope1:.2f}x + {intercept1:.2f}$, $R^2 = {r1**2:.4f}$\n\n'
        f'{label2}:\n'
        f'$y = {slope2:.2f}x + {intercept2:.2f}$, $R^2 = {r2**2:.4f}$'
    )
    ax.text(0.05, 0.80, regression_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=infobox_props, zorder=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    full_title = f"{title}\n{USER_ID}"
    ax.set_title(full_title, fontweight='bold', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Δημιουργήθηκε το γράφημα: {save_path}")

# --- 3. Κύρια Συνάρτηση Δημιουργίας Γραφικών (για το Lab 3) ---

def generate_lab3_plots():
    output_dir = "lab3_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Ενότητα Α: Προενισχυτής (Ερώτημα 1) ---
    
    # 1. Σύγκριση V_o(pin 3) και V_o(pin 4)
    range_pin3 = (0.0, 1.1)
    range_pin4 = (-1.1, 0.0) 
    
    plot_comparison(vi_A, vo_pin3_A, "Vo (pin 3)",
                    vi_A, vo_pin4_A, "Vo (pin 4)",
                    "Τάση Εισόδου Vi (V)", "Τάση Εξόδου Vo (V)",
                    "Χαρακτηριστικές Προενισχυτή (Ενότητα Α)",
                    os.path.join(output_dir, "plot_A1_pins.png"),
                    linear_range1=range_pin3,
                    linear_range2=range_pin4,
                    show_annotations=True) # <-- ΕΝΕΡΓΟΠΟΙΗΜΕΝΟ

    # 2. Χαρακτηριστική V_i = f[V_o(pin 4-3)]
    range_diff = (-11.5, 11.5) 
    
    plot_characteristic(vo_diff_A, vi_A, "Τάση Εξόδου Vo (pin 4-3) (V)", "Τάση Εισόδου Vi (V)",
                        "Χαρακτηριστική Προενισχυτή (Ενότητα Α)",
                        "Δεδομένα (Πίνακας Α)", os.path.join(output_dir, "plot_A1_diff.png"),
                        linear_range=range_diff,
                        show_annotations=True) # <-- ΕΝΕΡΓΟΠΟΙΗΜΕΝΟ

    # --- Ενότητα Β: Σήμα Σφάλματος (Ερώτημα 4) ---
    
    # 3. Διάγραμμα από Πίνακα 2
    range_B2 = (-120, 120)
    plot_characteristic(angle_in_B2, vo_B2, "Γωνία Εισόδου V2 (μοίρες)", "Τάση Εξόδου Vo (V)",
                        "Χαρακτηριστική Σήματος Σφάλματος ($\\theta_{out} = 0^o$)",
                        "Δεδομένα (Πίνακας 2)", os.path.join(output_dir, "plot_B4_P2.png"),
                        linear_range=range_B2,
                        show_annotations=True) 
    
    # 4. Διάγραμμα από Πίνακα 3
    range_B3 = (-150, 60)
    plot_characteristic(angle_in_B3, vo_B3, "Γωνία Εισόδου V2 (μοίρες)", "Τάση Εξόδου Vo (V)",
                        "Χαρακτηριστική Σήματος Σφάλματος ($\\theta_{out} = -60^o$)",
                        "Δεδομένα (Πίνακας 3)", os.path.join(output_dir, "plot_B4_P3.png"),
                        linear_range=range_B3,
                        show_annotations=True) 

    # --- Ενότητα Γ: Ηλεκτρικό Σύστημα (Ερώτημα 4) ---
    
    # 5. Διάγραμμα Κέρδους vs Νεκρής Ζώνης (Πίνακας 5)
    plot_characteristic(gain_C5, deadband_total_C5, "Κέρδος (K)", "Συνολική Νεκρή Ζώνη (μοίρες)",
                        "Επίδραση Κέρδους στη Νεκρή Ζώνη (Ενότητα Γ)",
                        "Δεδομένα (Πίνακας 5)", os.path.join(output_dir, "plot_C4_Deadband.png"),
                        show_regression=False,
                        show_annotations=True) 

# --- 4. Εκτέλεση Κώδικα ---
if __name__ == "__main__":
    generate_lab3_plots()
    print("\n--- ΟΙ ΓΡΑΦΙΚΕΣ ΠΑΡΑΣΤΑΣΕΙΣ ΓΙΑ ΤΟ LAB 3 ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ ---")
    print(f"Θα τις βρείτε στον φάκελο '{os.path.abspath('lab3_plots')}'")
