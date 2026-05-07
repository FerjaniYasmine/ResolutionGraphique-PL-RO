import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def verifier(coeffs, signe, b, x):
    val = sum(a * xi for a, xi in zip(coeffs, x))
    if signe == "<=":  return val <= b + 1e-8
    if signe == ">=":  return val >= b - 1e-8
    return abs(val - b) < 1e-6

def resoudre_pl(c_obj, contraintes, mode):
    n = len(c_obj)

    hyperplans = []
    for coeffs, _, b in contraintes:
        hyperplans.append((list(coeffs), b))
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        hyperplans.append((e, 0.0))   # xi = 0

    sommets = []
    for combo in combinations(range(len(hyperplans)), n):
        A = [hyperplans[k][0] for k in combo]
        b = [hyperplans[k][1] for k in combo]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if all(xi >= -1e-8 for xi in x):
            valide = all(
                verifier(coeffs, signe, bv, x)
                for coeffs, signe, bv in contraintes
            )
            if valide:
                sommets.append(tuple(max(0, xi) for xi in x))

    if not sommets:
        return None, None, []

    unique = []
    for s in sommets:
        if not any(np.allclose(s, u, atol=1e-6) for u in unique):
            unique.append(s)

    best, best_z = None, (-1e18 if mode == "max" else 1e18)
    for x in unique:
        z = sum(c * xi for c, xi in zip(c_obj, x))
        if (mode == "max" and z > best_z) or (mode == "min" and z < best_z):
            best_z, best = z, x

    return best, best_z, unique

def tracer_graphique(c_obj, contraintes, sommets, best, mode):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    x_max = max((s[0] for s in sommets), default=10) * 1.5 + 5
    y_max = max((s[1] for s in sommets), default=10) * 1.5 + 5
    xs = np.linspace(0, x_max, 500)

    for i, (coeffs, signe, b) in enumerate(contraintes):
        a1, a2 = coeffs
        if abs(a2) > 1e-10:
            ys = (b - a1 * xs) / a2
            ax.plot(xs, ys, color=colors[i % 10],
                    label=f"C{i+1}: {a1}x₁ + {a2}x₂ {signe} {b}", linewidth=1.5)
        elif abs(a1) > 1e-10:
            xv = b / a1
            ax.axvline(xv, color=colors[i % 10],
                       label=f"C{i+1}: {a1}x₁ {signe} {b}", linewidth=1.5)

    try:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        pts = sorted(sommets, key=lambda p: np.arctan2(
            p[1] - np.mean([s[1] for s in sommets]),
            p[0] - np.mean([s[0] for s in sommets])))
        poly = Polygon(pts, closed=True)
        ax.add_patch(plt.Polygon(pts, alpha=0.15, color='steelblue'))
    except Exception:
        pass

    for s in sommets:
        ax.plot(s[0], s[1], 'o', color='gray', markersize=6)

    # Point optimal
    ax.plot(best[0], best[1], '*', color='red', markersize=14,
            label=f"Optimal ({best[0]:.2f}, {best[1]:.2f})")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"PL — {mode.upper()}  Z = {sum(c*xi for c,xi in zip(c_obj,best)):.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

class PLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PL — Résolution Graphique Généralisée")
        self.geometry("680x700")
        self.configure(bg="#f4f4f4")
        self.resizable(True, True)

        self.nb_vars   = tk.IntVar(value=2)
        self.nb_ctrs   = tk.IntVar(value=2)
        self.type_opt  = tk.StringVar(value="max")

        self.obj_entries   = []
        self.ctr_entries   = []
        self.ctr_signes    = []
        self.ctr_rhs       = []

        self._build_static_ui()
        self._build_dynamic_ui()

    def _build_static_ui(self):
        tk.Label(self, text="Programmation Linéaire",
                 font=("Arial", 17, "bold"), bg="#f4f4f4").pack(pady=(12, 4))

        top = tk.Frame(self, bg="#f4f4f4")
        top.pack(fill="x", padx=14, pady=4)

        frv = tk.LabelFrame(top, text="Variables de décision", padx=8, pady=6, bg="#f4f4f4")
        frv.pack(side="left", padx=(0, 8))
        tk.Spinbox(frv, from_=2, to=10, textvariable=self.nb_vars,
                   width=4, command=self._build_dynamic_ui).pack()

        frc = tk.LabelFrame(top, text="Nombre de contraintes", padx=8, pady=6, bg="#f4f4f4")
        frc.pack(side="left", padx=(0, 8))
        tk.Spinbox(frc, from_=1, to=20, textvariable=self.nb_ctrs,
                   width=4, command=self._build_dynamic_ui).pack()

        frt = tk.LabelFrame(top, text="Optimisation", padx=8, pady=6, bg="#f4f4f4")
        frt.pack(side="left")
        tk.Radiobutton(frt, text="Maximiser", variable=self.type_opt,
                       value="max", bg="#f4f4f4").pack(anchor="w")
        tk.Radiobutton(frt, text="Minimiser", variable=self.type_opt,
                       value="min", bg="#f4f4f4").pack(anchor="w")

        tk.Button(self, text="▶  Résoudre", command=self._resoudre,
                  bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                  relief="flat", padx=16, pady=6).pack(pady=8)

        self.result_label = tk.Label(self, text="", font=("Arial", 11, "bold"),
                                     fg="#1a5276", bg="#f4f4f4", wraplength=640, justify="left")
        self.result_label.pack(pady=(0, 6), padx=14, anchor="w")

    def _build_dynamic_ui(self, *_):
        if hasattr(self, "_scroll_frame"):
            self._scroll_frame.destroy()

        n = self.nb_vars.get()
        m = self.nb_ctrs.get()

        # Canvas + scrollbar
        container = tk.Frame(self, bg="#f4f4f4")
        container.pack(fill="both", expand=True, padx=14, pady=4)
        self._scroll_frame = container

        canvas = tk.Canvas(container, bg="#f4f4f4", highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#f4f4f4")
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        self.obj_entries  = []
        self.ctr_entries  = []
        self.ctr_signes   = []
        self.ctr_rhs      = []

        fr_obj = tk.LabelFrame(inner, text="Fonction objectif  Z =",
                               padx=8, pady=8, bg="#f4f4f4")
        fr_obj.pack(fill="x", padx=4, pady=6)

        row_obj = tk.Frame(fr_obj, bg="#f4f4f4")
        row_obj.pack(anchor="w")
        for i in range(n):
            if i > 0:
                tk.Label(row_obj, text=" + ", bg="#f4f4f4").pack(side="left")
            e = tk.Entry(row_obj, width=6)
            e.insert(0, str(40 if i == 0 else 30))
            e.pack(side="left")
            tk.Label(row_obj, text=f" x{i+1}", bg="#f4f4f4",
                     font=("Arial", 10)).pack(side="left")
            self.obj_entries.append(e)

        fr_ctr = tk.LabelFrame(inner, text="Contraintes",
                               padx=8, pady=8, bg="#f4f4f4")
        fr_ctr.pack(fill="x", padx=4, pady=6)

        default_coefs = [[1, 1], [2, 1]]
        default_rhs   = [100, 150]

        for c in range(m):
            row = tk.Frame(fr_ctr, bg="#f4f4f4")
            row.pack(anchor="w", pady=3)

            tk.Label(row, text=f"C{c+1}: ", bg="#f4f4f4",
                     font=("Arial", 9, "bold"), width=4).pack(side="left")

            coef_row = []
            for i in range(n):
                if i > 0:
                    tk.Label(row, text=" + ", bg="#f4f4f4").pack(side="left")
                e = tk.Entry(row, width=6)
                val = (default_coefs[c][i]
                       if c < len(default_coefs) and i < len(default_coefs[c])
                       else 1)
                e.insert(0, str(val))
                e.pack(side="left")
                tk.Label(row, text=f" x{i+1}", bg="#f4f4f4",
                         font=("Arial", 10)).pack(side="left")
                coef_row.append(e)
            self.ctr_entries.append(coef_row)

            sv = tk.StringVar(value="<=")
            om = tk.OptionMenu(row, sv, "<=", ">=", "=")
            om.config(width=3)
            om.pack(side="left", padx=4)
            self.ctr_signes.append(sv)

            e_rhs = tk.Entry(row, width=7)
            e_rhs.insert(0, str(default_rhs[c] if c < len(default_rhs) else 100))
            e_rhs.pack(side="left")
            self.ctr_rhs.append(e_rhs)

        tk.Label(inner,
                 text="  x₁, x₂, …, xₙ ≥ 0  (contraintes de non-négativité implicites)",
                 bg="#f4f4f4", fg="#555", font=("Arial", 9, "italic")).pack(anchor="w", pady=(4, 0))


    def _resoudre(self):
        n = self.nb_vars.get()
        m = self.nb_ctrs.get()
        mode = self.type_opt.get()

        try:
            c_obj = [float(e.get()) for e in self.obj_entries]
            contraintes = []
            for c in range(m):
                coeffs = [float(self.ctr_entries[c][i].get()) for i in range(n)]
                signe  = self.ctr_signes[c].get()
                b      = float(self.ctr_rhs[c].get())
                contraintes.append((coeffs, signe, b))
        except ValueError:
            messagebox.showerror("Erreur", "Certaines valeurs ne sont pas des nombres valides.")
            return

        best, best_z, sommets = resoudre_pl(c_obj, contraintes, mode)

        if best is None:
            messagebox.showwarning("Pas de solution", "Aucun sommet réalisable trouvé.\nVérifiez vos contraintes.")
            self.result_label.config(text="Aucune solution réalisable.")
            return

        vars_str = "   ".join(f"x{i+1} = {best[i]:.4f}" for i in range(n))
        self.result_label.config(
            text=f"{'MAX' if mode=='max' else 'MIN'}  Z = {best_z:.4f}\n{vars_str}"
        )

        if n == 2:
            tracer_graphique(c_obj, contraintes, sommets, best, mode)
        else:
            messagebox.showinfo(
                "Résultat",
                f"{'Maximisation' if mode=='max' else 'Minimisation'} terminée.\n\n"
                f"Z = {best_z:.4f}\n\n" +
                "\n".join(f"x{i+1} = {best[i]:.4f}" for i in range(n)) +
                "\n\n(Graphique disponible uniquement pour 2 variables)"
            )



if __name__ == "__main__":
    app = PLApp()
    app.mainloop()