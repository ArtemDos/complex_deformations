import numpy as np

class AnalyticalDerivatives:
    """
    Класс для расчета аналитических производных вектора Ильюшина
    на основе параметрических уравнений винтовой траектории.
    """
    def __init__(self, c=25e-4, a=157e-4, n_turns=2.0, duration=50.0, nu=0.29):
        """
        Args:
            c (float): Радиус цилиндра траектории.
            a (float): Шаг винтовой линии.
            n_turns (float): Количество витков (оборотов) на 2-м этапе.
            duration (float): Длительность 2-го этапа в условных единицах времени (шагах).
            nu (float): Коэффициент Пуассона (0.5 для несжимаемого).
        """
        self.c = c
        self.a = a
        self.nu = nu
        # Угловая скорость omega = полный угол / время
        self.omega = (2 * np.pi * n_turns) / duration 
    

        if abs(nu - 0.5) < 1e-6:
             self.k_vol = 0.0
        else:
            # Коэффициент сжимаемости для плоского напряженного состояния
            # k_vol = (1 - 2v) / 3(1 - v)
            self.k_vol = (1 - 2 * nu) / (3 * (1 - nu))
        
    def get_derivatives(self, time_local):
        """
        Возвращает 1-ю, 2-ю и 3-ю аналитические производные для компонент вектора Ильюшина.
        
        Args:
            time_local (np.array): Массив времени от начала 2-го этапа (начинается с 0).
            
        Returns:
            d1 (3 x N), d2 (3 x N), d3 (3 x N)
            где d1[0] - это dEps1/dt, d1[1] - dEps2/dt и т.д.
        """

        # Текущий угол поворота alpha = omega * t
        alpha = self.omega * time_local

        c = self.c
        a = self.a
        w = self.omega
        sqrt3 = np.sqrt(3)

        A_theta = (sqrt3 * a * w) / (4 * np.pi)
        
        # eps_z'
        d1_ez = -c * np.sin(alpha) * w

        # eps_theta'
        d1_et = A_theta + 0.5 * c * np.sin(alpha) * w
        
        # eps_theta_z'
        d1_etz = (sqrt3 / 2) * c * np.cos(alpha) * w

        # eps_mean' = k_vol * (eps_z' + eps_theta')
        d1_mean = self.k_vol * (d1_ez + d1_et)

        # eps_z''
        d2_ez = -c * (w**2) * np.cos(alpha)

        # eps_theta''
        d2_et = 0.0 + 0.5 * c * (w**2) * np.cos(alpha)

        # eps_theta_z''
        d2_etz = -(sqrt3 / 2) * c * (w**2) * np.sin(alpha)
        
        # eps_mean''
        d2_mean = self.k_vol * (d2_ez + d2_et)

        # eps_z'''
        d3_ez = c * (w**3) * np.sin(alpha)

        # eps_theta'''
        d3_et = -0.5 * c * (w**3) * np.sin(alpha)

        # eps_theta_z'''
        d3_etz = -(sqrt3 / 2) * c * (w**3) * np.cos(alpha)
        
        # eps_mean'''
        d3_mean = self.k_vol * (d3_ez + d3_et)

        def build_vector(dez, det, detz, dmean):
            v1 = dez - dmean                            # Ni_1 = eps_z - eps_mean
            v2 = (1/sqrt3) * (dez + 2*det - 3*dmean)    # Ni_2 = (1/sqrt3) * (eps_z + 2*eps_theta - 3*eps_mean)
            v3 = (2/sqrt3) * detz                       # Ni_3 = (2/sqrt3) * eps_theta_z
            return np.array([v1, v2, v3])


        v_d1 = build_vector(d1_ez, d1_et, d1_etz, d1_mean)
        v_d2 = build_vector(d2_ez, d2_et, d2_etz, d2_mean)
        v_d3 = build_vector(d3_ez, d3_et, d3_etz, d3_mean)
        
        return v_d1, v_d2, v_d3
