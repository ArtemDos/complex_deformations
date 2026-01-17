import numpy as np

class AnalyticalDerivatives:
    """
    Класс для расчета аналитических производных вектора Ильюшина
    на основе параметрических уравнений винтовой траектории.
    """
    def __init__(self, c=25e-4, a=157e-4, n_turns=2.0, duration=50.0):
        """
        Args:
            c (float): Радиус цилиндра траектории.
            a (float): Шаг винтовой линии.
            n_turns (float): Количество витков (оборотов) на 2-м этапе.
            duration (float): Длительность 2-го этапа в условных единицах времени (шагах).
        """
        self.c = c
        self.a = a
        # Угловая скорость omega = Полный угол / Время
        self.omega = (2 * np.pi * n_turns) / duration 
        
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

        # d(Eps_Z)/dt
        d_ez = -c * np.sin(alpha) * w
        
        # d(Eps_Theta)/dt
        d_et = ( (sqrt3 * a) / (4 * np.pi) ) * w + 0.5 * c * np.sin(alpha) * w
        
        # d(Eps_ThetaZ)/dt (сдвиг)
        d_etz = (sqrt3 / 2) * c * np.cos(alpha) * w
        
        # d(Ni_1)/dt = d(Eps_Z)/dt
        dE1 = d_ez
        
        # d(Ni_2)/dt = (1/sqrt3) * (d(Eps_Z)/dt + 2 * d(Eps_Theta)/dt)
        dE2 = (1 / sqrt3) * (d_ez + 2 * d_et)
        
        # d(Ni_3)/dt = (2/sqrt3) * d(Eps_ThetaZ)/dt
        dE3 = (2 / sqrt3) * d_etz
        
        return dE1, dE2, dE3
