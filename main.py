import numpy as np

def act(x):
    return 0 if x < 0.5 else 1 # Функция активации

def go(house, rock, attr): # Функция для входного сигнала
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0] # Вес для первого нейрона
    w12 = [0.4, -0.5, 1] # Вес для второго нейрона
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1]) # Вектор связи

    sum_hidden = np.dot(weight1, x) # Сумма для скрытых нейронов
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden]) # Сумму пропускаем через функцию активации
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: "+str(y))

    return y

house = 1
rock = 0
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")

