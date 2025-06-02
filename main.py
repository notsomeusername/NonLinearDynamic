import sys
import numpy as np
import random
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

# --- Начальные условия ---

def ic_homogeneous(N):
    delta_1, delta_2 = 0.8, 1.6
    r0 = np.array([delta_1 * random.random() + 0.7 for _ in range(N)])
    phi0 = np.array([(2 * np.pi * 3 * (i) / 40 + delta_2 * random.random()) % (2*np.pi) for i in range(N)])
    return r0, phi0

def ic_spatial_disorder(N):
    r0 = np.random.uniform(0, 1, N)
    phi0 = np.random.uniform(0, 2*np.pi, N)
    return r0, phi0

def ic_inhomogeneous(N):
    m = 2
    phi_0 = 1
    r0 = np.array([random.random() for _ in range(N)])
    # phi0 = np.array([((2*np.pi*m)/(N*(i+1)) + phi_0) % (2*np.pi) for i in range(N)])
    phi0 = np.array([(2 * np.pi * 2 * (i) / 40 + 1.6 * random.random()) % (2*np.pi) for i in range(N)])
    return r0, phi0

# Пример: добавьте свои функции так:
# def ic_all_ones(N):
#     r0 = np.ones(N)
#     phi0 = np.zeros(N)
#     return r0, phi0

IC_TYPES = {
    "Пространственный беспорядок": ic_spatial_disorder,
    "Простарнственно однородные волны": ic_homogeneous,
    "Простарнственно неоднородные волны": ic_inhomogeneous,
    # "Все амплитуды 1, фазы 0": ic_all_ones,
}

# --- Модель ---

def F(r, a=11.5):
    return 2 * a * r**5 - a * r**3 + r

def oscillator_rhs_polar(t, y, N, u, d, a):
    r = y[:N]
    phi = y[N:]
    drdt = np.zeros(N)
    dphidt = np.zeros(N)

    for j in range(N):
        r_j = r[j]
        phi_j = phi[j]
        r_left = r[j - 1] if j > 0 else 0
        phi_left = phi[j - 1] if j > 0 else 0
        r_right = r[j + 1] if j < N - 1 else 0
        phi_right = phi[j + 1] if j < N - 1 else 0

        drdt[j] = (u / 2) * (
            -F(r_j, a)
            + d * (
                r_left * np.cos(phi_j - phi_left)
                - 2 * r_j
                + r_right * np.cos(phi_right - phi_j)
            )
        )

        if r_j != 0:
            dphidt[j] = (u * d / 2) * (
                r_right / r_j * np.sin(phi_right - phi_j)
                - r_left / r_j * np.sin(phi_j - phi_left)
            )

    return np.concatenate([drdt, dphidt])

def simulate_polar_chain(N, u, d, a, t_max, dt=0.05, ic_func=ic_spatial_disorder):
    t_eval = np.arange(0, t_max, dt)
    r0, phi0 = ic_func(N)
    y0 = np.concatenate([r0, phi0])

    sol = solve_ivp(
        fun=lambda t, y: oscillator_rhs_polar(t, y, N, u, d, a),
        t_span=(0, t_max),
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )

    r_sol = sol.y[:N].T
    phi_sol = sol.y[N:].T
    return r_sol, phi_sol, sol.t

def find_closest_index(array, target):
    return np.argmin(np.abs(array - target))

# --- Виджет для графиков ---

class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=6, dpi=100, dark=True):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if dark:
            self.fig.patch.set_facecolor('#222222')
        super().__init__(self.fig)

# --- Главное окно ---

class OscillatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscillator Chain Simulation")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #222222; color: #dddddd;")
        self.init_ui()
        self.r = None
        self.phi = None
        self.t = None
        self.N = None

    def init_ui(self):
        # --- Параметры ---
        param_box = QtWidgets.QGroupBox("Параметры")
        param_layout = QtWidgets.QFormLayout()
        self.u_edit = QtWidgets.QLineEdit("0.01")
        self.d_edit = QtWidgets.QLineEdit("0.6")
        self.a_edit = QtWidgets.QLineEdit("11.5")
        self.tmax_edit = QtWidgets.QLineEdit("3000")
        self.N_edit = QtWidgets.QLineEdit("40")
        for w in [self.u_edit, self.d_edit, self.a_edit, self.tmax_edit, self.N_edit]:
            w.setMaximumWidth(80)
            w.setStyleSheet("background-color: #333333; color: #dddddd;")
        param_layout.addRow("u:", self.u_edit)
        param_layout.addRow("d:", self.d_edit)
        param_layout.addRow("a:", self.a_edit)
        param_layout.addRow("t_max:", self.tmax_edit)
        param_layout.addRow("N:", self.N_edit)

        # --- Тип начальных условий ---
        self.ic_combo = QtWidgets.QComboBox()
        for name in IC_TYPES.keys():
            self.ic_combo.addItem(name)
        self.ic_combo.setStyleSheet("background-color: #333333; color: #dddddd;")
        param_layout.addRow("Начальные условия:", self.ic_combo)

        param_box.setLayout(param_layout)

        # --- Кнопка запуска ---
        self.run_btn = QtWidgets.QPushButton("Построить графики")
        self.run_btn.setStyleSheet("background-color: #444444; color: #dddddd;")
        self.run_btn.clicked.connect(self.run_simulation)

        # --- Слайдер времени ---
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_update)
        self.slider_label = QtWidgets.QLabel("Время: 0.00 c")
        self.slider_label.setStyleSheet("color: #dddddd;")

        # --- Графики ---
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444444; }
            QTabBar::tab { background: #333333; color: #dddddd; padding: 8px; }
            QTabBar::tab:selected { background: #444444; }
        """)

        # Статические графики
        self.static_canvas = MplCanvas(width=12, height=6, dark=True)
        self.tabs.addTab(self.static_canvas, "Статические графики")

        # Динамические графики
        self.dyn_canvas = MplCanvas(width=12, height=6, dark=True)
        self.tabs.addTab(self.dyn_canvas, "Динамика")

        # --- Layout ---
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(param_box)
        left_layout.addWidget(self.run_btn)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.slider_label)
        left_layout.addWidget(self.slider)
        left_layout.addStretch()

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(left_layout, 0)
        main_layout.addWidget(self.tabs, 1)

    def run_simulation(self):
        try:
            u = float(self.u_edit.text())
            d = float(self.d_edit.text())
            a = float(self.a_edit.text())
            t_max = float(self.tmax_edit.text())
            self.N = int(self.N_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения.")
            return

        ic_name = self.ic_combo.currentText()
        ic_func = IC_TYPES[ic_name]

        self.r, self.phi, self.t = simulate_polar_chain(self.N, u, d, a, t_max, ic_func=ic_func)
        self.update_static_plots(t_max)
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.t)-1)
        self.slider.setValue(0)
        self.slider_update(0)

    def update_static_plots(self, t_max):
        self.static_canvas.fig.clf()
        gs = self.static_canvas.fig.add_gridspec(2, 3, wspace=0.35, hspace=0.35)
        times = [0, t_max / 2, t_max]
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        titles = [f"t={t:.1f} c" for t in times]

        for i, (t_point, color, title) in enumerate(zip(times, colors, titles)):
            idx = find_closest_index(self.t, t_point)
            # Амплитуда
            ax_a = self.static_canvas.fig.add_subplot(gs[0, i])
            ax_a.plot(np.arange(self.N), self.r[idx], color=color, marker='o', linestyle='-')
            ax_a.set_title(f"Аплитуда\n{title}", color='#dddddd', fontsize=11, pad=12)
            ax_a.set_xlabel("Номер осциллятора", color='#dddddd')
            ax_a.set_ylabel("Амплитуда", color='#dddddd')
            ax_a.grid(True, color='#444444')
            ax_a.set_facecolor('#222222')
            ax_a.tick_params(colors='#bbbbbb')
            for spine in ax_a.spines.values():
                spine.set_color('#888888')
            ax_a.set_ylim(0, 1.2)
            # Фаза
            ax_p = self.static_canvas.fig.add_subplot(gs[1, i])
            ax_p.plot(np.arange(self.N), self.phi[idx], color=color, marker='o', linestyle='-')
            ax_p.set_title(f"Фаза\n{title}", color='#dddddd', fontsize=11, pad=12)
            ax_p.set_xlabel("Номер осциллятора", color='#dddddd')
            ax_p.set_ylabel("Фаза", color='#dddddd')
            ax_p.grid(True, color='#444444')
            ax_p.set_facecolor('#222222')
            ax_p.tick_params(colors='#bbbbbb')
            for spine in ax_p.spines.values():
                spine.set_color('#888888')
            phi_min = np.min(self.phi)
            phi_max = np.max(self.phi)
            ax_p.set_ylim(phi_min - 0.1 * abs(phi_max - phi_min), phi_max + 0.1 * abs(phi_max - phi_min))

        self.static_canvas.fig.tight_layout()
        self.static_canvas.draw()

    def slider_update(self, idx):
        if self.r is None or self.phi is None or self.t is None:
            return
        idx = int(idx)
        self.slider_label.setText(f"Время: {self.t[idx]:.2f} c")
        self.update_slider_plots(idx)

    def update_slider_plots(self, idx):
        self.dyn_canvas.fig.clf()
        gs = self.dyn_canvas.fig.add_gridspec(2, 1, hspace=0.35)
        # Амплитуда
        ax_amp = self.dyn_canvas.fig.add_subplot(gs[0, 0])
        ax_amp.plot(np.arange(self.N), self.r[idx], 'o-', color='#1f77b4')
        ax_amp.set_title(f"Аплитуда осцилляторов, t={self.t[idx]:.2f} c", color='#dddddd', fontsize=12, pad=12)
        ax_amp.set_xlabel("Номер осциллятора", color='#dddddd')
        ax_amp.set_ylabel("Амплитуда", color='#dddddd')
        ax_amp.set_ylim(0, 1.2)
        ax_amp.grid(True, color='#444444')
        ax_amp.set_facecolor('#222222')
        ax_amp.tick_params(colors='#bbbbbb')
        for spine in ax_amp.spines.values():
            spine.set_color('#888888')
        # Фаза
        ax_phi = self.dyn_canvas.fig.add_subplot(gs[1, 0])
        ax_phi.plot(np.arange(self.N), self.phi[idx], 'o-', color='#d62728')
        ax_phi.set_title(f"Фаза осцилляторов, t={self.t[idx]:.2f} c", color='#dddddd', fontsize=12, pad=12)
        ax_phi.set_xlabel("Номер осциллятора", color='#dddddd')
        ax_phi.set_ylabel("Фаза", color='#dddddd')
        phi_min = np.min(self.phi)
        phi_max = np.max(self.phi)
        ax_phi.set_ylim(phi_min - 0.1 * abs(phi_max - phi_min), phi_max + 0.1 * abs(phi_max - phi_min))
        ax_phi.grid(True, color='#444444')
        ax_phi.set_facecolor('#222222')
        ax_phi.tick_params(colors='#bbbbbb')
        for spine in ax_phi.spines.values():
            spine.set_color('#888888')

        self.dyn_canvas.fig.tight_layout()
        self.dyn_canvas.draw()

# --- Тёмная тема для PyQt ---

def set_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(34,34,34))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220,220,220))
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(40,40,40))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(34,34,34))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(220,220,220))
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(220,220,220))
    dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220,220,220))
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(40,40,40))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220,220,220))
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

# --- Запуск ---

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    set_dark_theme(app)
    window = OscillatorApp()
    window.show()
    sys.exit(app.exec_())