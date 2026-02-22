import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def main():
    sns.set_theme(style="darkgrid", palette="deep")
    plt.rcParams['figure.figsize'] = (11, 7)

    np.random.seed(9876)

    sample_size = 25
    data_sample = np.random.exponential(scale=1.0, size=sample_size)

    med_val = np.median(data_sample)
    data_range = np.max(data_sample) - np.min(data_sample)

    mu_hat = np.mean(data_sample)
    centered = data_sample - mu_hat
    moment_2 = np.mean(centered ** 2)
    moment_3 = np.mean(centered ** 3)
    skewness_val = moment_3 / (moment_2 ** 1.5) if moment_2 != 0 else 0

    freq_dict = Counter(data_sample)
    max_freq = max(freq_dict.values())
    mode_val = [k for k, v in freq_dict.items() if v == max_freq][0]

    print("-" * 30)
    print(f"Выборочные статистики (n={sample_size}):")
    print(f"Мода:               {mode_val:.5f}")
    print(f"Медиана:            {med_val:.5f}")
    print(f"Размах:             {data_range:.5f}")
    print(f"Коэфф. асимметрии:  {skewness_val:.5f}")
    print("-" * 30)

    sorted_data = np.sort(data_sample)
    ecdf_y = np.arange(1, sample_size + 1) / sample_size

    plt.figure()
    plt.step(sorted_data, ecdf_y, where='post', label='ЭФР (выборка)', color='darkmagenta', linewidth=2)
    if sorted_data[0] > 0:
        plt.step([0, sorted_data[0]], [0, 0], where='post', color='darkmagenta', linewidth=2)

    x_grid = np.linspace(0, np.max(data_sample) * 1.1, 200)
    cdf_theory = stats.expon.cdf(x_grid, scale=1.0)
    plt.plot(x_grid, cdf_theory, color='black', linestyle='-.', label='Теоретическая ФР', alpha=0.7)

    plt.title("Эмпирическая и теоретическая функции распределения")
    plt.xlabel("X")
    plt.ylabel("F(X)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    bins_count = int(1 + np.log2(sample_size)) + 1
    sns.histplot(data_sample, stat='density', bins=bins_count, color='teal', label='Гистограмма', alpha=0.6)

    pdf_theory = stats.expon.pdf(x_grid, scale=1.0)
    plt.plot(x_grid, pdf_theory, color='crimson', linestyle='--', linewidth=2.5, label='Плотность Exp(1)')

    plt.title("Гистограмма выборки и истинная плотность")
    plt.xlabel("X")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data_sample, color='coral', fliersize=6)
    plt.title("Диаграмма размаха (Boxplot)")
    plt.xlabel("Значения выборки")
    plt.tight_layout()
    plt.show()

    n_iters = 1200

    boot_means = [np.mean(np.random.choice(data_sample, size=sample_size, replace=True)) for _ in range(n_iters)]

    plt.figure()
    sns.histplot(boot_means, kde=True, stat="density", color="mediumpurple", label="Бутстрап средних", alpha=0.6)

    clt_mu = 1.0
    clt_sigma = np.sqrt(1.0 / sample_size)

    x_mean_grid = np.linspace(min(boot_means) * 0.8, max(boot_means) * 1.2, 200)
    clt_pdf = stats.norm.pdf(x_mean_grid, loc=clt_mu, scale=clt_sigma)

    plt.plot(x_mean_grid, clt_pdf, color='darkred', linewidth=2.5, label="Оценка по ЦПТ")
    plt.title("Сравнение распределения среднего: Бутстрап и ЦПТ")
    plt.xlabel("Выборочное среднее")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def calc_skewness(arr):
        m = np.mean(arr)
        centered_arr = arr - m
        v2 = np.mean(centered_arr ** 2)
        v3 = np.mean(centered_arr ** 3)
        return v3 / (v2 ** 1.5) if v2 > 0 else 0.0

    boot_skewness = np.array(
        [calc_skewness(np.random.choice(data_sample, size=sample_size, replace=True)) for _ in range(n_iters)])

    prob_skew_less_1 = np.mean(boot_skewness < 1.0)
    print(f"Бутстрап-оценка вероятности P(Skewness < 1): {prob_skew_less_1:.4f}")

    plt.figure()
    sns.histplot(boot_skewness, kde=True, stat="density", color="goldenrod", label="Распределение асимметрии",
                 alpha=0.6)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=2, label='Граница x = 1')
    plt.title(f"Бутстраповская оценка плотности коэффициента асимметрии\nP(Skew < 1) ≈ {prob_skew_less_1:.4f}")
    plt.xlabel("Коэффициент асимметрии")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.show()

    boot_medians = [np.median(np.random.choice(data_sample, size=sample_size, replace=True)) for _ in range(n_iters)]

    k_med = sample_size // 2 + 1

    def exp_pdf(x): return np.exp(-x)

    def exp_cdf(x): return 1 - np.exp(-x)

    def exact_median_pdf(x):
        combinations = sp.comb(sample_size - 1, k_med - 1)
        return sample_size * exp_pdf(x) * combinations * (exp_cdf(x) ** (k_med - 1)) * (
                    (1 - exp_cdf(x)) ** (sample_size - k_med))

    plt.figure()
    sns.histplot(boot_medians, kde=False, stat="density", color="forestgreen", label="Бутстрап медианы", alpha=0.5,
                 bins=15)

    x_med_grid = np.linspace(min(boot_medians) * 0.7, max(boot_medians) * 1.2, 200)
    y_med_exact = [exact_median_pdf(val) for val in x_med_grid]

    plt.plot(x_med_grid, y_med_exact, color='darkorange', linewidth=2.5, label="Точная плотность медианы")
    plt.title("Распределение медианы выборки: Бутстрап vs Точная формула")
    plt.xlabel("Медиана")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()