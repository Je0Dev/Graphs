import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import os
import pandas as pd

# --- 0. Στοιχεία Φοιτητή (Όπως στο αρχείο-πρότυπο) ---
USER_ID = "$23044-23122$"

# Εφαρμογή του επαγγελματικού θέματος (theme) του seaborn
# Ακριβώς όπως στο 'plots-better.py'
sns.set_theme(style="whitegrid", palette="deep")

# --- 1. Στυλ Πλαισίου Πληροφοριών (Όπως στο αρχείο-πρότυπο) ---
infobox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)


# --- 2. Βοηθητικές Συναρτήσεις Γραφικών (Προσαρμοσμένες από το 'plots-better.py') ---

def plot_characteristic(x, y, x_label, y_label, title, data_label, save_path, show_regression=True):
    """
    Δημιουργεί ένα γράφηma, προσαρμοσμένο από το plots-better.py.
    Τροποποίηση: Επιστρέφει fig, ax για περαιτέρω επεξεργασία.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    x_min, x_max = np.min(x), np.max(x)
    text_x_pos = 0.05 if x_min >= 0 else 0.65 

    ax.plot(x, y, 'o-', label=data_label, zorder=5) # 'o-' = Κύκλοι + Γραμμή
    
    # Λογική μορφοποίησης annotations από το αρχείο-πρότυπο
    # Τροποποίηση: Προστέθηκε το "kΩ" για να πιάνει και δεκαδικά
    for i in range(len(x)):
        if "Ρεύμα" in y_label or "kΩ" in y_label:
            label_text = f'({x[i]:.2f}, {y[i]:.2f})'
        else: # Ταχύτητα, Αντίσταση (Ω)
            label_text = f'({x[i]:.2f}, {y[i]:.0f})'
        
        ax.annotate(label_text, (x[i], y[i]), textcoords="offset points", 
                    xytext=(0,5), ha='center', fontsize=7.5, alpha=0.6)
    
    if show_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
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
    
    # Τροποποίηση: Δεν κλείνουμε το plot, το επιστρέφουμε
    # plt.savefig(save_path, dpi=200) 
    # plt.close(fig)
    return fig, ax


def plot_comparison(x1, y1, label1, x2, y2, label2, x_label, y_label, title, save_path, show_regression=True):
    """
    Δημιουργεί ένα γράφημα σύγκρισης, προσαρμοσμένο από το plots-better.py.
    Τροποποίηση: Επιστρέφει fig, ax και έχει σημαία show_regression.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # --- Δεδομένα 1 ---
    ax.plot(x1, y1, 'o-', label=label1, alpha=0.8, zorder=5)
    # Λογική μορφοποίησης annotations από το αρχείο-πρότυπο
    # Τροποποίηση: Προστέθηκε το "kΩ" για να πιάνει και δεκαδικά
    for i in range(len(x1)):
        if "Ρεύμα" in y_label or "kΩ" in y_label: label_text = f'({x1[i]:.2f}, {y1[i]:.2f})'
        else: label_text = f'({x1[i]:.2f}, {y1[i]:.0f})'
        ax.annotate(label_text, (x1[i], y1[i]), textcoords="offset points", 
                    xytext=(0,5), ha='center', fontsize=7.5, alpha=0.6, color=ax.lines[-1].get_color())

    # --- Δεδομένα 2 ---
    ax.plot(x2, y2, 's-', label=label2, alpha=0.8, zorder=5) # 's-' = Τετράγωνα
    for i in range(len(x2)):
        if "Ρεύμα" in y_label or "kΩ" in y_label: label_text = f'({x2[i]:.2f}, {y2[i]:.2f})'
        else: label_text = f'({x2[i]:.2f}, {y2[i]:.0f})'
        ax.annotate(label_text, (x2[i], y2[i]), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=7.5, alpha=0.6, color=ax.lines[-1].get_color())

    if show_regression:
        # Παλινδρόμηση 1
        slope1, intercept1, r1, _, _ = stats.linregress(x1, y1)
        x1_fit_line = np.linspace(np.min(x1), np.max(x1), 100)
        y_fit1 = slope1 * x1_fit_line + intercept1
        ax.plot(x1_fit_line, y_fit1, linestyle='--', label=f'Παλινδρόμηση ({label1})', zorder=10, linewidth=2, color=ax.lines[-2].get_color())
        
        # Παλινδρόμηση 2
        slope2, intercept2, r2, _, _ = stats.linregress(x2, y2)
        x2_fit_line = np.linspace(np.min(x2), np.max(x2), 100)
        y_fit2 = slope2 * x2_fit_line + intercept2
        ax.plot(x2_fit_line, y_fit2, linestyle=':', label=f'Παλινδρόμηση ({label2})', zorder=10, linewidth=2, color=ax.lines[-1].get_color())
        
        # Κείμενο για το infobox
        regression_text = (
            f'{label1}:\n'
            f'$y = {slope1:.2f}x + {intercept1:.2f}$, $R^2 = {r1**2:.4f}$\n\n'
            f'{label2}:\n'
            f'$y = {slope2:.2f}x + {intercept2:.2f}$, $R^2 = {r2**2:.4f}$'
        )
        ax.text(0.05, 0.05, regression_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=infobox_props, zorder=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    full_title = f"{title}\n{USER_ID}"
    ax.set_title(full_title, fontweight='bold', fontsize=15)
    
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    
    # Τροποποίηση: Δεν κλείνουμε το plot, το επιστρέφουμε
    # plt.savefig(save_path, dpi=200)
    # plt.close(fig)
    return fig, ax

# --- 3. Κύρια Συνάρτηση Δημιουργίας Γραφικών (Εργασία 3) ---

def generate_thermistor_plots():
    output_dir = "thermistor_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Δεδομένα για το Διάγραμμα 1 (Αυτοθέρμανση) ---
    data_air = {
        'P (mW)': [9.7, 37, 216.4, 353.4], # Διορθωμένη τιμή
        'R (kΩ)': [2.3, 2.2, 1.53, 1.02]
    }
    df_air = pd.DataFrame(data_air)

    data_water = {
        'P (mW)': [3.45, 34.03, 164, 320],
        'R (kΩ)': [2.17, 1.87, 1.62, 1.41]
    }
    df_water = pd.DataFrame(data_water)

    # --- Δεδομένα για το Διάγραμμα 2 (Βαθμονόμηση) ---
    data_calib = {
        'Θερμοκρασία (°C)': [25, 30, 37, 42, 48, 53, 59, 65, 70, 78, 83, 5.5],
        'Αντίσταση (Ω)': [2200, 2013, 1430, 1101, 1002, 810, 610, 510, 407, 320, 290, 5100]
    }
    df_calib = pd.DataFrame(data_calib).sort_values(by='Θερμοκρασία (°C)')
    R_unknown = 533 # Αντίσταση (Ω)

    # --- ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ ---

    print("Δημιουργία Γραφήματος 1 (Αυτοθέρμανση)...")
    save_path_1 = os.path.join(output_dir, "thermistor_plot_01_self_heating.png")
    
    fig1, ax1 = plot_comparison(
        x1 = df_air['P (mW)'], y1 = df_air['R (kΩ)'], label1 = "Στον Αέρα (Πίν. 1)",
        x2 = df_water['P (mW)'], y2 = df_water['R (kΩ)'], label2 = "Στο Νερό (Πίν. 1)",
        x_label = "Ισχύς P (mW) [Λογαριθμικός Άξονας]",
        y_label = "Αντίσταση R (kΩ)",
        title = "Διάγραμμα 1: Φαινόμενο Αυτοθέρμανσης (R = f(P))",
        save_path = save_path_1,
        show_regression = False  # Η παλινδρόμηση δεν έχει νόημα σε log scale
    )
    
    # Ειδική προσαρμογή για την Εργασία 3: Log scale
    ax1.set(xscale="log")
    ax1.grid(True, which="both", ls="--") # Προσθήκη grid και για τον log άξονα
    
    fig1.savefig(save_path_1, dpi=200)
    plt.close(fig1)
    print(f"Δημιουργήθηκε το γράφημα: {save_path_1}")

    # ---
    
    print("Δημιουργία Γραφήματος 2 (Βαθμονόμηση)...")
    save_path_2 = os.path.join(output_dir, "thermistor_plot_02_calibration.png")
    
    fig2, ax2 = plot_characteristic(
        x = df_calib['Θερμοκρασία (°C)'],
        y = df_calib['Αντίσταση (Ω)'],
        x_label = "Θερμοκρασία θ (°C)",
        y_label = "Αντίσταση R (Ω)",
        title = "Διάγραμμα 2: Καμπύλη Βαθμονόμησης Θερμίστορ (R = f(θ))",
        data_label = "Σημεία Βαθμονόμησης (Πίν. 2)",
        save_path = save_path_2,
        show_regression = False # Η σχέση είναι εκθετική, όχι γραμμική
    )
    
    # Ειδική προσαρμογή για την Εργασία 3: Παρεμβολή (Interpolation)
    interp_linear = interp1d([510, 610], [65, 59]) # Γραμμική παρεμβολή μεταξύ (65C, 510Ω) και (59C, 610Ω)
    T_unknown = interp_linear(R_unknown)
    
    # Σχεδίαση του άγνωστου σημείου και των γραμμών του
    ax2.plot(T_unknown, R_unknown, 'r*', markersize=15, zorder=10, 
             label=f'Άγνωστο Σώμα ({T_unknown:.1f} °C, {R_unknown} Ω)')
    ax2.hlines(R_unknown, 0, T_unknown, colors='red', linestyles='dotted', zorder=9)
    ax2.vlines(T_unknown, 0, R_unknown, colors='red', linestyles='dotted', zorder=9)
    
    # Ρύθμιση ορίων και επαν-ενεργοποίηση του legend
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 5500)
    ax2.legend(loc='best', fontsize=11)
    
    fig2.savefig(save_path_2, dpi=200)
    plt.close(fig2)
    print(f"Δημιουργήθηκε το γράφημα: {save_path_2}")


# --- 4. Εκτέλεση Κώδικα ---
if __name__ == "__main__":
    generate_thermistor_plots()
    print("\n--- ΟΛΕΣ ΟΙ ΓΡΑΦΙΚΕΣ ΠΑΡΑΣΤΑΣΕΙΣ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ ---")
    print(f"Θα τις βρείτε στον φάκελο '{os.path.abspath('thermistor_plots')}'")
