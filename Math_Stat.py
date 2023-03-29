import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from PIL import Image

# Параметры полной выборки.
glob_nums = []
glob_fun_vib = []
glob_num_vib = []
glob_norm_func = []
glob_prop_norm = []
glob_prop_vib = []
Disp = 0
Sigma = 0
glob_quantity = 0
Mu = 0
data = []

# Параметры для вырезки из выборки.
data_slice = []
Disp_slice = 0
glob_nums_slice = []
glob_num_vib_slice = []
Mu_slice = 0
Sigma_slice = 0
glob_prop_vib_slice = []
glob_fun_vib_slice = []
glob_prop_norm_slice = []
glob_norm_func_slice = []

# Функция для создания распределения в нормальной форме.
def build_norm(x, Mu, Sigma):
    return 1/(Sigma * np.sqrt(2 * np.pi)) * np.exp(-(pow(x-Mu, 2)/(2*pow(Sigma, 2))))

# Основная функция, которая генерирует выборку и вырезку из выборки.
def build_selection(average, disp, quantity, Size_of_slice):
    average = float(average)
    sigma = np.sqrt(float(disp))

    global glob_nums, glob_fun_vib, glob_num_vib, glob_norm_func, glob_prop_norm, glob_quantity, Mu, glob_prop_vib, Sigma, Disp, data, estimated_norm, nums

    global data_slice, Disp_slice, glob_nums_slice, glob_num_vib_slice, Mu_slice, Sigma_slice, glob_prop_vib_slice, glob_norm_func_slice

    glob_quantity = quantity
    data = np.array(sorted(list(map(int,np.random.normal(average,  sigma, quantity)))))

    nums = np.unique(data)
    Disp = np.var(data)
    glob_nums, glob_num_vib = np.unique(data, return_counts=True)
    Mu = np.mean(data)
    Sigma = np.std(data)
    glob_prop_vib = np.array([i/len(data) for i in glob_num_vib])
    glob_fun_vib = np.cumsum(glob_prop_vib)
    glob_prop_norm = build_norm(glob_nums, Mu, Sigma)
    glob_norm_func = np.cumsum(glob_prop_norm)
    
    # Вырезка:
    data_slice = np.random.choice(data, Size_of_slice, replace=True)
    Disp_slice = np.var(data_slice)
    glob_nums_slice, glob_num_vib_slice = np.unique(data_slice, return_counts=True)
    Mu_slice = np.mean(data_slice)
    Sigma_slice = np.std(data_slice)
    glob_prop_vib_slice = np.array([i/len(data_slice) for i in glob_num_vib_slice])
    glob_fun_vib_slice = np.cumsum(glob_prop_vib_slice)
    glob_prop_norm_slice = build_norm(glob_nums_slice, Mu_slice, Sigma_slice)
    glob_norm_func_slice = np.cumsum(glob_prop_norm_slice)

    estimated_norm = np.array(glob_prop_norm) * quantity


def Kolmogorov_criterion(alpha):
    # Вывод графика функции выборки
    plt.figure(figsize=(10, 5))
    plt.title("Функция выборки")
    plt.xticks(glob_nums)
    plt.plot(glob_nums, glob_fun_vib)
    plt.savefig("image/Func.png")
    func = Image.open("image/Func.png")

    # Вывод графика функции предполагаемого нормалного распределения
    plt.figure(figsize=(10, 5))
    plt.title("Функция нормального распределения")
    plt.xticks(glob_nums)
    plt.plot(glob_nums, glob_norm_func)
    plt.savefig("image/Fun_norm.png")
    func_norm = Image.open("image/Fun_norm.png")

    # Вывод графика сравнения распределений (Предполагаемое нормальное и выборка)
    plt.figure(figsize=(10, 5))
    plt.title("Сравнение распределений")
    plt.xticks(glob_nums)
    plt.scatter(glob_nums, glob_fun_vib, linewidths=4, marker="_", c="red", label='Распределение выборки')
    plt.plot(glob_nums, glob_norm_func, label='Распределение нормальное')
    plt.legend()
    plt.savefig("image/dis_comparison.png")
    dis_comparison = Image.open("image/dis_comparison.png")

    lambda_crit = {
        0.15:1.1380,
        0.10:1.2238,
        0.05:1.3581,
        0.025:1.4802,
        0.02:1.5174,
        0.01:1.6276,
        0.005:1.7308,
        0.001:1.9495,
    }

    step_of_freedom = len(nums)

    Dn = max(abs(np.array(glob_fun_vib) - np.array(glob_norm_func)))

    if step_of_freedom >= 40:
        simga_N_slash = (Dn) * np.sqrt(step_of_freedom)
    else:
        simga_N = (Dn) * np.sqrt(step_of_freedom)
        simga_N_slash = simga_N * (1 + 0.12 / np.sqrt(step_of_freedom) + 0.11 / step_of_freedom)

    alpha = float(alpha)
    text = f"Максимальное расхождение функций - {round(Dn, 5)} \nЛямбда - {round(simga_N_slash, 5)} \nЛямбда критическая - {lambda_crit[alpha]}\n"
    text += 'Гипотеза Н0 подтверждена' if lambda_crit[alpha] > simga_N_slash else 'Гипотеза Н1 подтверждена'

    return [func, func_norm, dis_comparison, text]

def Pirson_criterion(alpha):
    # Вывод графика плотности выборки
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-notebook')
    plt.title("График плотности выборки")
    plt.xticks(glob_nums)
    plt.bar(glob_nums, glob_num_vib)
    plt.savefig("image/Cons.png")
    cons = Image.open("image/Cons.png")

    # Вывод графика плотности предполагаемого нормального распределения
    plt.figure(figsize=(10, 5))
    plt.title("Плотность нормального распределения")
    plt.xticks(glob_nums)
    plt.plot(glob_nums, glob_prop_norm)
    plt.savefig("image/Cons_norm.png")
    func_cons_norm = Image.open("image/Cons_norm.png")

    # Вывод графика сравнения плотностей (Предполагаемое нормальное и выборка)
    plt.figure(figsize=(10, 5))
    plt.xticks(glob_nums)
    plt.title("Сравнение плотностей")
    plt.plot(glob_nums, estimated_norm, label='Плотность нормального распределения', c="r")
    plt.bar(glob_nums, glob_num_vib, label='Плотность выборки')
    plt.legend()
    plt.savefig("image/cons_comparison.png")
    cons_comparison = Image.open("image/cons_comparison.png")


    X2 = []
    tmp = pow((np.array(glob_num_vib) - (np.array(estimated_norm))), 2)

    for q,x in enumerate(estimated_norm):
        X2.append(tmp[q] / x)

    step_of_freedom = len(nums) - 3
    alpha = float(alpha)
    Hi_crit = scipy.stats.chi2.ppf(1-alpha, step_of_freedom)
    
    text = f"Хи квадрат - {round(sum(X2), 5)} \nХи квадрат критический - {round(Hi_crit, 5)}\n"
    text += 'Гипотеза Н0 подтверждена' if Hi_crit > sum(X2) else 'Гипотеза Н1 подтверждена'

     
    return [cons, func_cons_norm, cons_comparison, text]

# Доверитеьный интервал для мат. ожидания (используется вся выборка)
def Mu_common(alpha):
    global glob_nums
    global glob_prop_vib
    global Sigma
    alpha = float(alpha)

    l = Mu - ((scipy.stats.norm.ppf(1-alpha)*Sigma)/np.sqrt(len(glob_nums)))
    r = Mu + ((scipy.stats.norm.ppf(1-alpha)*Sigma)/np.sqrt(len(glob_nums)))

    plt.figure(figsize=(10, 5))
    plt.xticks(glob_nums)
    plt.title(f"Доверительная вероятность = {1-alpha}")
    plt.bar(glob_nums, glob_prop_vib, label='Плотность выборки')
    plt.bar([l, r], 0.15, width=0.2, label=f"Доверительный интревал [{l}; {r}]")
    plt.legend()
    plt.savefig("image/mu_c.png")
    mu_c = Image.open("image/mu_c.png")

    return l, r, mu_c

# Доверитеьный интервал для мат. ожидания, использующий вырезку из выборки. 
# Получается востановленный интервал, немного отличающийся от обычного
# (использует распределение по Стьюденту)
def Mu_student(alpha):
    global glob_prop_vib_slice, glob_nums_slice, Disp_slice, data_slice, Mu_slice
    alpha = float(alpha)

    s = np.sqrt(len(glob_nums_slice)/(len(glob_nums_slice)-1) * Disp_slice)
    l = Mu_slice - (max(scipy.stats.t.interval(1-alpha, len(glob_nums_slice)-1))*s)/np.sqrt(len(glob_nums_slice))
    r = Mu_slice + (max(scipy.stats.t.interval(1-alpha, len(glob_nums_slice)-1))*s)/np.sqrt(len(glob_nums_slice))

    plt.figure(figsize=(10, 5))
    plt.xticks(glob_nums_slice)
    plt.title(f"Доверительная вероятность = {1-alpha}")
    plt.bar(glob_nums_slice, glob_prop_vib_slice, label='Плотность выборки')
    plt.bar([l, r], 0.15, width=0.2, label=f"Доверительный интревал [{l}; {r}]")
    plt.legend()
    plt.savefig("image/mu_student.png")
    mu_student = Image.open("image/mu_student.png")

    return l, r, mu_student

# Доверительный интервал для Дисперсии (Хи квадрат)
def D_hi2(alpha):
    global glob_prop_vib_slice, glob_nums_slice, Disp_slice, data_slice, Mu_slice
    alpha = float(alpha)

    s = len(glob_nums_slice)/(len(glob_nums_slice)-1) * Disp_slice
    l = ((len(glob_nums_slice)-1)*s)/(scipy.stats.chi2.ppf(alpha, len(glob_nums_slice)))
    r = ((len(glob_nums_slice)-1)*s)/(scipy.stats.chi2.ppf(1-alpha, len(glob_nums_slice)))

    plt.figure(figsize=(10, 5))
    plt.xticks(glob_nums_slice)
    plt.bar(glob_nums_slice, glob_prop_vib_slice, label='Плотность выборки')
    plt.bar([Mu_slice - np.sqrt(l), Mu + np.sqrt(l)], 0.15, width=0.2, label="Доверительный интревал")
    plt.bar([Mu_slice - np.sqrt(r), Mu + np.sqrt(r)], 0.15, width=0.2, label="Доверительный интревал")
    plt.legend()
    plt.savefig("image/disp_c.png")
    disp_c = Image.open("image/disp_c.png")

    return l, r, disp_c

with gr.Blocks() as demo:
    gr.Markdown("ИДЗ №3")
    with gr.Tab("Создание выборки"):
        # Генерирование выборки по параметрам + генерирование вырезки из выборки
        average_input = gr.Textbox(label="Среднее значние выборки")
        disp_input = gr.Textbox(label="Дисперсия")
        quantity_input = gr.Slider(0, 2000, label="Количество элементов")
        Silce_size = gr.Slider(1, 100, step=1, label="Вырезка из выборки (в %)")
        graphs_button = gr.Button("Создать выборку")

        graphs_button.click(build_selection, inputs=[average_input, disp_input, quantity_input, Silce_size], outputs=[])
        
    with gr.Tab("Критерий Колмогорова"):
        a_input = gr.Textbox(label="Критерий значимости")

        graphs_button = gr.Button("Проверить по критерию Колмогорова")

        func_output = gr.Image(label="График распределения выборки")
        func_norm_output = gr.Image(label="График функции нормального распределения")
        dis_comparison_output = gr.Image(label="Сравнение распределений")
        output = gr.Textbox(label="Вывод")

        graphs_button.click(Kolmogorov_criterion, inputs=[a_input], outputs=[func_output, func_norm_output, dis_comparison_output, output])

        
    with gr.Tab("Критерий Пирсона"):
        a_input = gr.Textbox(label="Критерий значимости")

        graphs_button = gr.Button("Проверить по критерию Пирсона")

        cons_output = gr.Image(label="График плотности выборки")
        func_cons_norm_output = gr.Image(label="График плотности нормального распределения")
        cons_comparison = gr.Image(label="Сравнение плотностей")
        
        output = gr.Textbox(label="Вывод")

        graphs_button.click(Pirson_criterion, inputs=[a_input], outputs=[cons_output, func_cons_norm_output, cons_comparison, output])
            

    with gr.Tab("Доверительные интервалы"):
        a_input = gr.Textbox(label="Критерий значимости")

        # Доверительный интервал для мат. ожидания.
        mu_button = gr.Button("Доверительный интервал мат. ожидания")

        l = gr.Textbox(label="Левая граница дов. интервала для мат. ожидания")
        r = gr.Textbox(label="Правая граница дов. интервала для мат. ожидания")

        mu_inter = gr.Image(label="Доверительный интервал для мат. ожидания")
        mu_button.click(Mu_common, inputs=[a_input], outputs=[l, r, mu_inter])

        # Доверительный интервал для мат. ожидания по распределению Стьюдента (испоьзует часть выборки).
        mu_student_button = gr.Button("Доверительный интервал мат. ожидания по распределению Стьюдента")

        l_student = gr.Textbox(label="Левая граница дов. интервала для мат. ожидания (Стьюдент)")
        r_student = gr.Textbox(label="Правая граница дов. интервала для мат. ожидания (Стьюдент)")

        mu_inter_student = gr.Image(label="Доверительный интервал для мат. ожидания (по распределению Стьюдента)")
        mu_student_button.click(Mu_student, inputs=[a_input], outputs=[l_student, r_student, mu_inter_student])

        # Доверительный интервал для дисперсии.
        disp_inter_button = gr.Button("Доверительный интервал для дисперсии")

        r_disp = gr.Textbox(label="Левая граница дов. интервала для диспресии")
        l_disp = gr.Textbox(label="Правая граница дов. интервала для дисперсии")

        disp_inter_button.click(D_hi2, inputs=[a_input], outputs=[l_disp, r_disp])

demo.launch()