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
        
        # Коэффициент сжимаемости для плоского напряженного состояния
        # k_vol = (1 - 2v) / 3(1 - v)
        self.k_vol = (1 - 2 * nu) / (3 * (1 - nu))
        
    def get_derivatives(self, time_local):
        """
        Возвращает аналитические производные d(Eps)/dt для компонент вектора Ильюшина.
        
        Args:
            time_local (np.array): Массив времени от начала 2-го этапа (начинается с 0).
            
        Returns:
            tuple: (dEps1_dt, dEps2_dt, dEps3_dt)
        """

        # Текущий угол поворота alpha = omega * t
        alpha = self.omega * time_local

        c = self.c
        a = self.a
        w = self.omega
        sqrt3 = np.sqrt(3)
        
        # d(eps_z)/dt
        d_ez = -c * np.sin(alpha) * w
        
        # d(eps_theta)/dt
        d_et = ( (sqrt3 * a) / (4 * np.pi) ) * w + 0.5 * c * np.sin(alpha) * w
        
        # d(eps_theta_z)/dt
        d_etz = (sqrt3 / 2) * c * np.cos(alpha) * w

        # d(eps_mean)/dt = K_vol * (d_ez + d_et)
        d_mean = self.k_vol * (d_ez + d_et)

        # dNi_1 = d(e_z) = d_ez - d_mean
        dE1 = d_ez - d_mean
        
        # dNi_2 = (1/sqrt3) * (d_ez + 2*d_et - 3*d_mean)
        dE2 = (1 / sqrt3) * (d_ez + 2 * d_et - 3 * d_mean)
        
        # dNi_3 = (2/sqrt3) * d_etz
        dE3 = (2 / sqrt3) * d_etz
        
        return dE1, dE2, dE3