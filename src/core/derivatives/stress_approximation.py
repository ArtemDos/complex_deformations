import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class VolumetricModel:
    """
    Модель аппроксимации объемной деформации (или среднего напряжения)
    в зависимости от параметра винтовой траектории x.
    
    Модель: f(x) = p * cos(w*x + phi) + b1*x + b2
    """
    def __init__(self):
        self.params = None
        
    def _model_func(self, x, p, w, phi, b1, b2):
        return p * np.cos(w * x + phi) + b1 * x + b2
    
    def fit(self, x_data, y_data, guess_w=None):
        """
        Подбирает параметры модели под данные (x, y).
        """
        if guess_w is None:
            # Грубая оценка частоты: 2 полных колебания на весь путь
            guess_w = (4 * np.pi) / (x_data.max() - x_data.min())
            
        # Начальное приближение
        p0 = [
            (y_data.max() - y_data.min()) / 2, # Амплитуда
            guess_w,                           # Частота
            0.0,                               # Фаза
            (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]), # Тренд b1
            y_data.mean()                      # Сдвиг b2
        ]
        
        try:
            self.params, _ = curve_fit(self._model_func, x_data, y_data, p0=p0, maxfev=10000)
        except RuntimeError:
            print("Warning: VolumetricModel fitting failed, using initial guess.")
            self.params = p0
            
        return self.params
    
    def predict(self, s):
        if self.params is None:
            raise ValueError("Model is not fitted yet.")
        return self._model_func(s, *self.params)
    
    def get_derivatives_x(self, x):
        """
        Возвращает 1-ю, 2-ю и 3-ю производные функции по x: f'(x), f''(x), f'''(x).
        """
        if self.params is None:
            raise ValueError("Model is not fitted yet.")
            
        p, w, phi, b1, b2 = self.params
        arg = w * x + phi
        
        d1 = -p * w * np.sin(arg) + b1
        d2 = -p * (w**2) * np.cos(arg)
        d3 = p * (w**3) * np.sin(arg)
        
        return d1, d2, d3


class GeometricProperties:
    """
    Расчет геометрических характеристик (кривизна, кручение)
    с учетом поправки на сжимаемость из VolumetricModel.
    """

    _ANALYTIC_COLS = [
        "dEps_1_analytic",
        "dEps_2_analytic",
        "dEps_3_analytic",
        "d2Eps_1_analytic",
        "d2Eps_2_analytic",
        "d2Eps_3_analytic",
        "d3Eps_1_analytic",
        "d3Eps_3_analytic",
        "d3Eps_3_analytic",
    ]

    def __init__(self, volumetric_model, E=2.05e5, nu=0.29):
        self.vol_model = volumetric_model
        self.E = E
        self.nu = nu
        self.k_mat = (1 - 2 * nu) / E

    def _delta_derivatives(self, x_array):
        d1_sig, d2_sig, d3_sig = self.vol_model.get_derivatives_x(x_array)
        k = self.k_mat
        return d1_sig * k, d2_sig * k, d3_sig * k

    def _assemble_vaj(self, df, x_array):
        sqrt3 = np.sqrt(3)
        missing = [c for c in self._ANALYTIC_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame missing columns: {missing}")
    
        d1_delta, d2_delta, d3_delta = self._delta_derivatives(x_array)
        v1 = df["dEps_1_analytic"].to_numpy(dtype=float) - d1_delta
        a1 = df["d2Eps_1_analytic"].to_numpy(dtype=float) - d2_delta
        j1 = df["d3Eps_1_analytic"].to_numpy(dtype=float) - d3_delta
        v2 = df["dEps_2_analytic"].to_numpy(dtype=float) - sqrt3 * d1_delta
        a2 = df["d2Eps_2_analytic"].to_numpy(dtype=float) - sqrt3 * d2_delta
        j2 = df["d3Eps_2_analytic"].to_numpy(dtype=float) - sqrt3 * d3_delta
        v3 = df["dEps_3_analytic"].to_numpy(dtype=float)
        a3 = df["d2Eps_3_analytic"].to_numpy(dtype=float)
        j3 = df["d3Eps_3_analytic"].to_numpy(dtype=float)
        V = np.column_stack((v1, v2, v3))
        A = np.column_stack((a1, a2, a3))
        J = np.column_stack((j1, j2, j3))
        return V, A, J

    def compute_geometry(self, df, x_array):
        V, A, J = self._assemble_vaj(df, x_array)
        cross_va = np.cross(V, A)
        num_k = np.linalg.norm(cross_va, axis=1)
        den_k = np.linalg.norm(V, axis=1) ** 3
        kappa = num_k / den_k
        num_t = np.sum(cross_va * J, axis=1)
        den_t = np.linalg.norm(cross_va, axis=1) ** 2
        tau = num_t / den_t
        return kappa, tau

    def get_derivatives(self, df, x_array):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("get_derivatives expects a pandas.DataFrame.")
        sqrt3 = np.sqrt(3)
        for c in ("dEps_1_analytic", "dEps_2_analytic", "dEps_3_analytic"):
            if c not in df.columns:
                raise KeyError(f"DataFrame missing column {c!r}.")
        d1_delta, _, _ = self._delta_derivatives(x_array)
        v1 = df["dEps_1_analytic"].to_numpy(dtype=float) - d1_delta
        v2 = df["dEps_2_analytic"].to_numpy(dtype=float) - sqrt3 * d1_delta
        v3 = df["dEps_3_analytic"].to_numpy(dtype=float)
        return [v1, v2, v3]
