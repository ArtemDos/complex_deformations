import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class SineLayer(nn.Module):
    """
    Кастомный слой с синусоидальной активацией (SIREN).
    Очень хорош для представления периодических сигналов и их производных.
    """
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
    
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class NeuralDerivatives:
    def __init__(self, hidden_dim=128, lr=0.005, epochs=3000, device='cpu'):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.model = None
        
        # Параметры нормализации (MinMax scaling)
        self.t_min = None
        self.t_max = None
        self.y_mean = None
        self.y_std = None

    def _normalize_t(self, t):
        # Масштабируем время в диапазон [-1, 1] для лучшей сходимости
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def _denormalize_deriv(self, d_norm, order=1):
        # Восстанавливаем реальный масштаб производной
        # dy/dx = (dy_norm * y_std) / (dx_norm * scale_t)
        scale_t = 2.0 / (self.t_max - self.t_min)
        return (d_norm * self.y_std) * (scale_t ** order)

    def fit(self, t, y):
        """
        Обучает модель на переданных точках (t, y).
        """
        # 1. Нормализация данных
        self.t_min, self.t_max = t.min(), t.max()
        self.y_mean, self.y_std = y.mean(), y.std()
        
        # Защита от деления на ноль, если сигнал константа
        if self.y_std < 1e-9: self.y_std = 1.0
            
        t_norm = self._normalize_t(t)
        y_norm = (y - self.y_mean) / self.y_std
        
        # Конвертация в тензоры
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 2. Инициализация модели
        self.model = MLP(self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Sheduler для уменьшения LR (помогает точнее сойтись в конце)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
        loss_fn = nn.MSELoss()
        
        # 3. Цикл обучения
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(t_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Ранняя остановка, если обучились идеально (опционально)
            if loss.item() < 1e-7:
                break
                
    def predict(self, t):
        """
        Предсказывает значения (сглаживание).
        """
        self.model.eval()
        t_norm = self._normalize_t(t)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        
        with torch.no_grad():
            y_norm_pred = self.model(t_tensor).cpu().numpy().flatten()
            
        return y_norm_pred * self.y_std + self.y_mean

    def compute_derivatives(self, t):
        """
        Вычисляет 1-ю, 2-ю и 3-ю производные через Autograd.
        """
        self.model.eval()
        t_norm = self._normalize_t(t)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        t_tensor.requires_grad = True
        
        # Прямой проход
        y_pred = self.model(t_tensor)
        
        # --- 1-я производная ---
        grads_1 = torch.autograd.grad(
            outputs=y_pred, 
            inputs=t_tensor, 
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True # Важно для взятия второй производной
        )[0]
        
        # --- 2-я производная ---
        grads_2 = torch.autograd.grad(
            outputs=grads_1, 
            inputs=t_tensor, 
            grad_outputs=torch.ones_like(grads_1),
            create_graph=True # Важно для взятия третьей производной
        )[0]
        
        # --- 3-я производная ---
        grads_3 = torch.autograd.grad(
            outputs=grads_2, 
            inputs=t_tensor, 
            grad_outputs=torch.ones_like(grads_2),
            create_graph=False # Дальше не нужно
        )[0]
        
        # Конвертация в numpy и денормализация
        d1 = self._denormalize_deriv(grads_1.detach().cpu().numpy().flatten(), order=1)
        d2 = self._denormalize_deriv(grads_2.detach().cpu().numpy().flatten(), order=2)
        d3 = self._denormalize_deriv(grads_3.detach().cpu().numpy().flatten(), order=3)
        
        return d1, d2, d3
