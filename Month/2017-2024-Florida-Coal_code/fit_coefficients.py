# This is a placeholder for the fit_coefficients.py file.
"""
Created on 2024/07/15
地区为 Florida
燃烧类型为:Coal
时间为:2017/01/01-2024/04/30
作者: natur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from numpy import polyfit, poly1d
from scipy import optimize
from scipy.optimize import OptimizeWarning
import warnings
import re
import os

# 确保目录存在
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# 加载数据
data_path = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_result/fitting_curve_emission.csv'
data = pd.read_csv(data_path)

# 保存路径
csv_save_dir = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/time-fitting/CSV'
fig_save_dir = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/time-fitting/fitting_fig'
mse_save_dir = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/time-fitting/MSE'

# 提取特定机组的数据
units = data['unit'].unique()

# 提取系数
def extract_coefficients(expression):
    pattern = r"(-?\d+\.?\d*) x\*x \+ (-?\d+\.?\d*) x\+ (-?\d+\.?\d*)"
    match = re.findall(pattern, expression)
    if match:
        return list(map(float, match[0]))
    return [None, None, None]

# 离群值处理函数
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 初始化 FIT 和 MSE 列表
FIT = []
MSE = []

# 遍历每个机组
for unit in units:
    filtered_data = data[(data['unit'] == unit) & (data['year'].between(2017, 2022))]
    filtered_data[['a', 'b', 'k']] = filtered_data['Piecewise-Quad-Linear'].apply(lambda x: extract_coefficients(x) if pd.notna(x) else [None, None, None]).apply(pd.Series)
    filtered_data.dropna(subset=['a', 'b', 'k'], inplace=True)
    
    # 按时间排序数据
    filtered_data['date'] = pd.to_datetime(filtered_data[['year', 'month']].assign(day=1))
    filtered_data = filtered_data.sort_values(by='date')
    filtered_data['time'] = (filtered_data['date'] - filtered_data['date'].min()).dt.days / 30

    # 离群值处理
    filtered_data = remove_outliers(filtered_data, 'a')
    filtered_data = remove_outliers(filtered_data, 'b')
    filtered_data = remove_outliers(filtered_data, 'k')

    # 保存时间顺序的 a, b, k 数据
    csv_save_path = os.path.join(csv_save_dir, f'{unit}_coefficients.csv')
    filtered_data[['date', 'a', 'b', 'k']].to_csv(csv_save_path, index=False)
    
    x = filtered_data['time'].values
    x1 = np.linspace(min(x), max(x), 500)

    for coef in ['a', 'b', 'k']:
        y = filtered_data[coef].values

        # 检查数据点数量
        if len(x) < 2:
            print(f"Skipping unit {unit}_{coef} due to insufficient data points.")
            continue

        # 线性拟合
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coeff_linear = polyfit(x, y, 1)
            f_linear = poly1d(coeff_linear)
            if w:
                for warning in w:
                    print(f"Linear fitting warning for {unit}_{coef}: {warning.message}")

        mse_linear = mean_squared_error(y, f_linear(x))
        f_linear_str = f'{coeff_linear[0]:.6f} x + {coeff_linear[1]:.6f}'

        # 二次拟合
        if len(x) >= 3:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                coeff_quad = polyfit(x, y, 2)
                f_quad = poly1d(coeff_quad)
                if w:
                    for warning in w:
                        print(f"Quadratic fitting warning for {unit}_{coef}: {warning.message}")

            mse_quad = mean_squared_error(y, f_quad(x))
            f_quad_str = f'{coeff_quad[0]:.6f} x^2 + {coeff_quad[1]:.6f} x + {coeff_quad[2]:.6f}'
        else:
            mse_quad = np.inf
            f_quad_str = None

        # 三次拟合
        if len(x) >= 4:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                coeff_cubic = polyfit(x, y, 3)
                f_cubic = poly1d(coeff_cubic)
                if w:
                    for warning in w:
                        print(f"Cubic fitting warning for {unit}_{coef}: {warning.message}")

            mse_cubic = mean_squared_error(y, f_cubic(x))
            f_cubic_str = f'{coeff_cubic[0]:.6f} x^3 + {coeff_cubic[1]:.6f} x^2 + {coeff_cubic[2]:.6f} x + {coeff_cubic[3]:.6f}'
        else:
            mse_cubic = np.inf
            f_cubic_str = None

        # 分段线性拟合
        p_best_plinear = None
        if len(x) >= 5:
            def piecewise_linear(x, x0, y0, k1, k2):
                return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1*x + y0-k1*x0, 
                                                           lambda x: k2*x + y0-k2*x0])

            perr_min = np.inf
            for n in range(100):
                k = np.random.rand(4)*20
                try:
                    p, e = optimize.curve_fit(piecewise_linear, x, y, p0=k)
                except (RuntimeError, OptimizeWarning):
                    continue
                perr = np.sum(np.abs(y-piecewise_linear(x, *p)))
                if perr < perr_min:
                    perr_min = perr
                    p_best_plinear = p

            if p_best_plinear is not None:
                mse_plinear = mean_squared_error(y, piecewise_linear(x, *p_best_plinear))
                f_plinear1 = f'{p_best_plinear[2]:.6f} x + {p_best_plinear[1] - p_best_plinear[2] * p_best_plinear[0]:.6f}'
                f_plinear2 = f'{p_best_plinear[3]:.6f} x + {p_best_plinear[1] - p_best_plinear[3] * p_best_plinear[0]:.6f}'
                segment_point = (p_best_plinear[0], p_best_plinear[1])
                f_plinear_str = f"('{f_plinear1}', '{f_plinear2}', Segment Point: {segment_point})"
            else:
                mse_plinear = np.inf
                f_plinear_str = None
        else:
            mse_plinear = np.inf
            f_plinear_str = None

        # 分段二次加线性拟合
        p_best_pql = None
        if len(x) >= 6:
            def piecewise_quad_linear(x, x0, y0, a, b, k):
                return np.piecewise(x, [x < x0, x >= x0], [lambda x: a*x*x + b*x + y0 - a*x0*x0 - b*x0, 
                                                           lambda x: k*x + y0 - k*x0])

            perr_min = np.inf
            for n in range(100):
                k = np.random.rand(5)*20
                try:
                    p, e = optimize.curve_fit(piecewise_quad_linear, x, y, p0=k, maxfev=10000)
                except (RuntimeError, OptimizeWarning):
                    continue
                perr = np.sum(np.abs(y-piecewise_quad_linear(x, *p)))
                if perr < perr_min:
                    perr_min = perr
                    p_best_pql = p

            if p_best_pql is not None:
                mse_pql = mean_squared_error(y, piecewise_quad_linear(x, *p_best_pql))
                f_pql1 = f'{p_best_pql[2]:.6f} x^2 + {p_best_pql[3]:.6f} x + {p_best_pql[1] - p_best_pql[2] * p_best_pql[0]**2 - p_best_pql[3] * p_best_pql[0]:.6f}'
                f_pql2 = f'{p_best_pql[4]:.6f} x + {p_best_pql[1] - p_best_pql[4] * p_best_pql[0]:.6f}'
                segment_point_pql = (p_best_pql[0], p_best_pql[1])
                f_pql_str = f"('{f_pql1}', '{f_pql2}', Segment Point: {segment_point_pql})"
            else:
                mse_pql = np.inf
                f_pql_str = None
        else:
            mse_pql = np.inf
            f_pql_str = None

        # 分段三次加二次拟合
        p_best_pcq = None
        if len(x) >= 7:
            def piecewise_cubic_quad(x, x0, y0, a, b, c, m, n):
                return np.piecewise(x, [x < x0, x >= x0], [lambda x: a*x*x*x + b*x*x + y0 - a*x0*x0*x0 - b*x0*x0 - c*x0, 
                                                           lambda x: m*x*x + n*x + y0 - m*x0*x0 - n*x0])

            perr_min = np.inf
            for n in range(100):
                k = np.random.rand(7)*20
                try:
                    p, e = optimize.curve_fit(piecewise_cubic_quad, x, y, p0=k, maxfev=10000)
                except (RuntimeError, OptimizeWarning, TypeError):
                    continue
                perr = np.sum(np.abs(y-piecewise_cubic_quad(x, *p)))
                if perr < perr_min:
                    perr_min = perr
                    p_best_pcq = p

            if p_best_pcq is None:
                mse_pcq = np.inf
                f_pcq1 = f_pcq2 = segment_point_pcq = None
                f_pcq_str = None
            else:
                mse_pcq = mean_squared_error(y, piecewise_cubic_quad(x, *p_best_pcq))
                f_pcq1 = f'{p_best_pcq[2]:.6f} x^3 + {p_best_pcq[3]:.6f} x^2 + {p_best_pcq[4]:.6f} x + {p_best_pcq[1] - p_best_pcq[2] * p_best_pcq[0]**3 - p_best_pcq[3] * p_best_pcq[0]**2 - p_best_pcq[4] * p_best_pcq[0]:.6f}'
                f_pcq2 = f'{p_best_pcq[5]:.6f} x^2 + {p_best_pcq[6]:.6f} x + {p_best_pcq[1] - p_best_pcq[5] * p_best_pcq[0]**2 - p_best_pcq[6] * p_best_pcq[0]:.6f}'
                segment_point_pcq = (p_best_pcq[0], p_best_pcq[1])
                f_pcq_str = f"('{f_pcq1}', '{f_pcq2}', Segment Point: {segment_point_pcq})"
        else:
            mse_pcq = np.inf
            f_pcq_str = None

        # 随机森林回归
        rf = RandomForestRegressor()
        rf.fit(x.reshape(-1, 1), y)
        y_rf = rf.predict(x1.reshape(-1, 1))

        mse_rf = mean_squared_error(y, rf.predict(x.reshape(-1, 1)))
        f_rf_str = "Random Forest Regressor"

        # 保存拟合结果到 FIT 和 MSE 列表中
        FIT.append([f'{unit}_{coef}', f_linear_str, f_quad_str, f_cubic_str, f_plinear_str, f_pql_str, f_pcq_str, f_rf_str])
        MSE.append([f'{unit}_{coef}', mse_linear, mse_quad, mse_cubic, mse_plinear, mse_pql, mse_pcq, mse_rf])

        # 保存拟合图
        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x1, f_linear(x1), label='Linear', color='red')
        plt.plot(x1, f_quad(x1), label='Quadratic', color='blue')
        plt.plot(x1, f_cubic(x1), label='Cubic', color='green')
        if p_best_plinear is not None:
            plt.plot(x1, piecewise_linear(x1, *p_best_plinear[:4]), label='Piecewise-Linear', color='purple')
        if p_best_pql is not None:
            plt.plot(x1, piecewise_quad_linear(x1, *p_best_pql[:5]), label='Piecewise-Quad-Linear', color='orange')
        if p_best_pcq is not None:
            plt.plot(x1, piecewise_cubic_quad(x1, *p_best_pcq[:7]), label='Piecewise-Cubic-Quad', color='brown')
        plt.plot(x1, y_rf, label='RandomForest', color='pink')
        plt.xlabel('Time (months)')
        plt.ylabel(f'{coef} value')
        plt.title(f'{unit}_{coef} Fits')
        plt.legend()
        ensure_dir(fig_save_dir)
        plt.savefig(os.path.join(fig_save_dir, f'{unit}_{coef}_fits.png'))
        plt.close()

# 保存 FIT 和 MSE 列表到 CSV 文件
fit_df = pd.DataFrame(FIT, columns=['Unit', 'Linear', 'Quadratic', 'Cubic', 'Piecewise-Linear', 'Piecewise-Quad-Linear', 'Piecewise-Cubic-Quad', 'RandomForest'])
mse_df = pd.DataFrame(MSE, columns=['Unit', 'Linear', 'Quadratic', 'Cubic', 'Piecewise-Linear', 'Piecewise-Quad-Linear', 'Piecewise-Cubic-Quad', 'RandomForest'])

fit_df.to_csv(os.path.join(mse_save_dir, 'fitting_curve_emission.csv'), index=False)
mse_df.to_csv(os.path.join(mse_save_dir, 'MSE_results.csv'), index=False)

print("Fitting and saving completed.")
