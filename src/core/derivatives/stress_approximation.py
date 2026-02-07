import numpy as np
from scipy.optimize import curve_fit

class VolumetricModel:
    """
    Модель аппроксимации объемной деформации (или среднего напряжения)
    в зависимости от длины дуги s.
    
    Модель: f(s) = p * cos(w*s + phi) + b1*s + b2
    """
    def __init__(self):
        self.params = None
        
    def _model_func(self, s, p, w, phi, b1, b2):
        return p * np.cos(w * s + phi) + b1 * s + b2
    
    def fit(self, s_data, y_data, guess_w=None):
        """
        Подбирает параметры модели под данные (s, y).
        """
        if guess_w is None:
            # Грубая оценка частоты: 2 полных колебания на весь путь
            guess_w = (4 * np.pi) / (s_data.max() - s_data.min())
            
        # Начальное приближение
        p0 = [
            (y_data.max() - y_data.min()) / 2, # Амплитуда
            guess_w,                           # Частота
            0.0,                               # Фаза
            (y_data[-1] - y_data[0]) / (s_data[-1] - s_data[0]), # Тренд b1
            y_data.mean()                      # Сдвиг b2
        ]
        
        try:
            self.params, _ = curve_fit(self._model_func, s_data, y_data, p0=p0, maxfev=10000)
        except RuntimeError:
            print("Warning: VolumetricModel fitting failed, using initial guess.")
            self.params = p0
            
        return self.params
    
    def predict(self, s):
        if self.params is None:
            raise ValueError("Model is not fitted yet.")
        return self._model_func(s, *self.params)
    
    def get_derivatives_s(self, s):
        """
        Возвращает 1-ю и 2-ю производные функции по s: f'(s), f''(s).
        """
        if self.params is None:
            raise ValueError("Model is not fitted yet.")
            
        p, w, phi, b1, b2 = self.params
        arg = w * s + phi
        
        d1 = -p * w * np.sin(arg) + b1
        d2 = -p * (w**2) * np.cos(arg)
        
        return d1, d2


class SemiEmpiricalGeometry:
    """
    Расчет геометрических характеристик (кривизна, кручение)
    с учетом поправки на сжимаемость из VolumetricModel.
    """
    def __init__(self, volumetric_model, E=2.05e5, nu=0.29, c=25e-4, a=157e-4):
        self.vol_model = volumetric_model
        self.E = E
        self.nu = nu
        self.c = c
        self.a = a
        # Коэффициент перевода Sigma_mean -> Eps_mean
        self.k_mat = (1 - 2*nu) / E
        
    def compute_geometry(self, s_array):
        """
        Вычисляет kappa(s) и tau(s).
        """
        # 1. Параметры кинематики (идеальный винт)
        L_turn = np.sqrt(self.a**2 + (2 * np.pi * self.c)**2)
        lam = (2 * np.pi) / L_turn  # lambda = d(alpha)/ds
        
        arg_kin = lam * s_array
        sqrt3 = np.sqrt(3)
        
        # 2. Идеальные производные (Kinematic, несжимаемые)
        # dEps_ideal / ds
        d1_ez = -self.c * lam * np.sin(arg_kin)
        d2_ez = -self.c * lam**2 * np.cos(arg_kin)
        d3_ez =  self.c * lam**3 * np.sin(arg_kin)
        
        d1_etz = (sqrt3/2) * self.c * lam * np.cos(arg_kin)
        d2_etz = -(sqrt3/2) * self.c * lam**2 * np.sin(arg_kin)
        d3_etz = -(sqrt3/2) * self.c * lam**3 * np.cos(arg_kin)
        
        # Ni_2 ideal (линейная часть)
        d1_ni2_id = self.a / L_turn
        d2_ni2_id = 0.0
        d3_ni2_id = 0.0
        
        # 3. Производные поправки Delta (из модели напряжения)
        # Мы берем производные от Sigma_mean по s и умножаем на k_mat, чтобы получить d(Eps_mean)/ds
        d1_sig, d2_sig = self.vol_model.get_derivatives_s(s_array)
        
        # Нам нужна еще 3-я производная для кручения
        p, w, phi, b1, b2 = self.vol_model.params
        d3_sig = p * (w**3) * np.sin(w * s_array + phi)
        
        d1_delta = d1_sig * self.k_mat
        d2_delta = d2_sig * self.k_mat
        d3_delta = d3_sig * self.k_mat
        
        # 4. Сборка вектора Ильюшина
        # Ni_1 = eps_z - delta
        v1 = d1_ez - d1_delta
        a1 = d2_ez - d2_delta
        j1 = d3_ez - d3_delta
        
        # Ni_2 = Ni_2_ideal - sqrt(3)*delta
        v2 = d1_ni2_id - sqrt3 * d1_delta
        a2 = d2_ni2_id - sqrt3 * d2_delta
        j2 = d3_ni2_id - sqrt3 * d3_delta
        
        # Ni_3 = (2/sqrt3) * eps_theta_z (не зависит от delta)
        v3 = (2/sqrt3) * d1_etz
        a3 = (2/sqrt3) * d2_etz
        j3 = (2/sqrt3) * d3_etz

        V = np.column_stack((v1, v2, v3))
        A = np.column_stack((a1, a2, a3))
        J = np.column_stack((j1, j2, j3))
        
        # Кривизна
        cross_va = np.cross(V, A)
        num_k = np.linalg.norm(cross_va, axis=1)
        den_k = np.linalg.norm(V, axis=1)**3
        kappa = num_k / den_k
        
        # Кручение
        num_t = np.sum(cross_va * J, axis=1)
        den_t = np.linalg.norm(cross_va, axis=1)**2
        tau = num_t / den_t
        
        return kappa, tau
