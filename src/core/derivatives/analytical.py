import numpy as np

class AnalyticalDerivatives:
    """
    Класс для расчета аналитических производных вектора Ильюшина
    на основе параметрических уравнений винтовой траектории
    для НЕСЖИМАЕМОГО материала.
    """
    def __init__(self, c=25e-4, a=157e-4):
        """
        Args:
            c (float): Радиус цилиндра траектории.
            a (float): Шаг винтовой линии.
        """
        self.c = c
        self.a = a
        
        # Длина одного полного витка траектории (L_turn) - это
        # гипотенуза треугольника развертки с катетами a и 2*pi*c
        self.L_turn = np.sqrt(a**2 + (2 * np.pi * c)**2)

        self.lam = (2 * np.pi) / self.L_turn
        
    def get_all_derivatives(self, s_array):
        """
        Возвращает производные вектора Ильюшина по длине дуги s: 
        d/ds (скорость), d2/ds2 (кривизна), d3/ds3 (рывок).
        """
        # Угол поворота
        alpha = self.lam * s_array
        lam = self.lam
        c = self.c
        a = self.a

        # d(Ni_1)/ds = -c * lambda * sin(alpha)
        d1_v1 = -c * lam * np.sin(alpha)
        
        # d(Ni_2)/ds = a / L_turn = const
        d1_v2 = np.full_like(s_array, a / self.L_turn)
        
        # d(Ni_3)/ds = c * lambda * cos(alpha)
        d1_v3 = c * lam * np.cos(alpha)

        d2_v1 = -c * (lam**2) * np.cos(alpha)
        d2_v2 = np.zeros_like(s_array)
        d2_v3 = -c * (lam**2) * np.sin(alpha)

        d3_v1 = c * (lam**3) * np.sin(alpha)
        d3_v2 = np.zeros_like(s_array)
        d3_v3 = -c * (lam**3) * np.cos(alpha)

        v_d1 = np.array([d1_v1, d1_v2, d1_v3])
        v_d2 = np.array([d2_v1, d2_v2, d2_v3])
        v_d3 = np.array([d3_v1, d3_v2, d3_v3])
        
        return v_d1, v_d2, v_d3
    
    def get_derivatives(self, s_array):
        d1, _, _ = self.get_all_derivatives(s_array)
        return d1[0], d1[1], d1[2]
        
    def get_values(self, s_array, eps_z_start=0.0, eps_theta_start=0.0):
        """
        Возвращает значения компонент вектора Ильюшина (Ni_1, Ni_2, Ni_3).
        """
        alpha = self.lam * s_array
        c = self.c
        a = self.a
        sqrt3 = np.sqrt(3)

        # Ni_1 = eps_z (без вычета mean, так как mean=0)
        v1 = c * (np.cos(alpha) - 1) + eps_z_start

        # Ni_2 = 1/sqrt(3) * (v1 + 2*et)
        linear_term = (sqrt3 / 2) * a * (s_array / self.L_turn)
        et = linear_term - 0.5 * c * (np.cos(alpha) - 1) + eps_theta_start
        v2 = (1 / sqrt3) * (v1 + 2 * et)
        
        # Ni_3 = (2/sqrt3) * eps_theta_z
        etz = (sqrt3 / 2) * c * np.sin(alpha)
        v3 = (2 / sqrt3) * etz
        
        return v1, v2, v3
