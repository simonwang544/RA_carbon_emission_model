# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:55:51 2021

@author: natur
"""
# 机组碳排放曲线拟合
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from numpy import polyfit, poly1d
from scipy import optimize

# 创建目录函数
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# 参考文件路径
Ref_Path = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant/hourly-emissions-2020-florida.csv'
Ref_df = pd.read_csv(Ref_Path, encoding='ISO-8859-1')  # 指定编码

def save_to_file(file_name, contents):
    with open(file_name, 'a') as fh:
        fh.write(contents)

MSE = []
FIT = []
for f in ['P100', 'P500', 'P1000', 'PBig']:  # 包括PBig
    Folder_Path = f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/{f}'
    os.chdir(Folder_Path)
    file_list = os.listdir(Folder_Path)
    for i in range(0, len(file_list)):
        if file_list[i] == '.DS_Store':
            print(f"Skipping file {file_list[i]} due to missing columns.")
            continue
        # 读取文件时尝试多种编码
        try:
            df0 = pd.read_csv(Folder_Path + '/' + file_list[i], engine='c', encoding='utf-8-sig')  # 指定UTF-8-SIG编码
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                df0 = pd.read_csv(Folder_Path + '/' + file_list[i], engine='c', encoding='ISO-8859-1')
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"Skipping file {file_list[i]} due to encoding error or parser error: {e}")
                continue
        
        # 打印文件的列名
        ##print(f"Columns in file {file_list[i]}: {df0.columns}")

        # 假设文件包含特定列名
        expected_columns = ['Gross Load (MW)', 'DEF(ton/MW)', 'AEF(ton/MW)']
        if not all(column in df0.columns for column in expected_columns):
            print(f"Skipping file {file_list[i]} due to missing columns.")
            continue

        df1 = df0[expected_columns]
        df = df1[np.isfinite(df1).all(1)]
        x = df['Gross Load (MW)']
        y = df['DEF(ton/MW)']
        Aef = sum(x*y)/sum(x)
        Aef_EPA = df['AEF(ton/MW)'].iloc[0]  # 获取“AEF(ton/MW)”列的第一个值
        x1 = np.linspace(0, int(max(x)), int(max(x))*10)
        xlist = x.tolist()
        ylist = y.tolist()
        unitname = file_list[i].split(".")

        # 线性拟合
        coeff_linear = polyfit(x, y, 1)
        f_linear = poly1d(coeff_linear)
        print("Linear Fitting For %s: %s" % (str(unitname[0]), f_linear))
        plt.plot(x, y, 'bx')
        plt.plot(x1, f_linear(x1), 'r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Linear')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Linear.png')
        plt.close()

        mse_linear = mean_squared_error(y, f_linear(x))
        mape_linear = mean_absolute_percentage_error(y, f_linear(x))

        # 二次拟合
        coeff_quad = polyfit(x, y, 2)
        f_quad = poly1d(coeff_quad)
        print("Quadratic Fitting For %s: %s" % (str(unitname[0]), f_quad))
        plt.figure()
        plt.plot(x, y, 'bx')
        plt.plot(x1, f_quad(x1), 'r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Quad')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Quad.png')
        plt.close()

        mse_quad = mean_squared_error(y, f_quad(x))
        mape_quad = mean_absolute_percentage_error(y, f_quad(x))

        # 三次拟合
        coeff_cubic = polyfit(x, y, 3)
        f_cubic = poly1d(coeff_cubic)
        print("Cubic Fitting For %s: %s" % (str(unitname[0]), f_cubic))
        plt.figure()
        plt.plot(x, y, 'bx')
        plt.plot(x1, f_cubic(x1), 'r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Cubic')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Cubic.png')
        plt.close()

        mse_cubic = mean_squared_error(y, f_cubic(x))
        mape_cubic = mean_absolute_percentage_error(y, f_cubic(x))

        # 分段线性拟合
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1*x + y0-k1*x0, 
                                                       lambda x: k2*x + y0-k2*x0])

        perr_min = np.inf
        p_best = None
        for n in range(100):
            k = np.random.rand(4)*20
            p, e = optimize.curve_fit(piecewise_linear, xlist, ylist, p0=k)
            perr = np.sum(np.abs(y-piecewise_linear(xlist, *p)))
            if perr < perr_min:
                perr_min = perr
                p_best = p

        plt.figure()
        plt.plot(x, y, 'bx')
        plt.plot(x1, piecewise_linear(x1, *p_best), 'r')
        plt.scatter(p_best[0], p_best[1], s=80, c='r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Piecewise-Linear')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Piecewise-Linear.png')
        plt.close()

        f_plinear1 = '%f x+%f' % (p_best[2], p_best[1]-p_best[2]*p_best[0])
        f_plinear2 = '%f x+%f' % (p_best[3], p_best[1]-p_best[3]*p_best[0])
        print("Piecewise-Linear Fitting For %s: %s and %s" % (str(unitname[0]), f_plinear1, f_plinear2))
        mse_plinear = mean_squared_error(y, piecewise_linear(xlist, *p_best))
        mape_plinear = mean_absolute_percentage_error(y, piecewise_linear(xlist, *p_best))

        # 分段二次加线性拟合
        def piecewise_quad_linear(x, x0, y0, a, b, k):
            return np.piecewise(x, [x < x0, x >= x0], [lambda x: a*x*x + b*x + y0 - a*x0*x0 - b*x0, 
                                                       lambda x: k*x + y0 - k*x0])

        perr_min = np.inf
        p_best = None
        for n in range(100):
            k = np.random.rand(5)*20
            p, e = optimize.curve_fit(piecewise_quad_linear, xlist, ylist, p0=k, maxfev=10000)  # 增加maxfev值
            perr = np.sum(np.abs(y-piecewise_quad_linear(xlist, *p)))
            if perr < perr_min:
                perr_min = perr
                p_best = p

        plt.figure()
        plt.plot(x, y, 'bx')
        plt.plot(x1, piecewise_quad_linear(x1, *p_best), 'r')
        plt.scatter(p_best[0], p_best[1], s=80, c='r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Piecewise-QL')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Piecewise-QL.png')
        plt.close()

        f_pql1 = '%f x*x + %f x+ %f' % (p_best[2], p_best[3], p_best[1]-p_best[2]*p_best[0]*p_best[0]-p_best[3]*p_best[0])
        f_pql2 = '%f x + %f' % (p_best[4], p_best[1]-p_best[4]*p_best[0])
        print("Piecewise-Quadratic-Linear Fitting For %s: %s and %s" % (str(unitname[0]), f_pql1, f_pql2))
        mse_pql = mean_squared_error(y, piecewise_quad_linear(xlist, *p_best))
        mape_pql = mean_absolute_percentage_error(y, piecewise_quad_linear(xlist, *p_best))

        # 分段三次加二次拟合
        def piecewise_cubic_quad(x, x0, y0, a, b, c, m, n):
            return np.piecewise(x, [x < x0, x >= x0], [lambda x: a*x*x*x + b*x*x + y0 - a*x0*x0*x0 - b*x0*x0 - c*x0, 
                                                       lambda x: m*x*x + n*x + y0 - m*x0*x0 - n*x0])

        perr_min = np.inf
        p_best = None
        for n in range(100):
            k = np.random.rand(7)*20
            try:
                p, e = optimize.curve_fit(piecewise_cubic_quad, xlist, ylist, p0=k, maxfev=10000)  # 增加maxfev值
            except RuntimeError as err:
                print(f"RuntimeError: {err}")
                continue
            perr = np.sum(np.abs(y-piecewise_cubic_quad(xlist, *p)))
            if perr < perr_min:
                perr_min = perr
                p_best = p

        if p_best is None:
            print(f"Skipping unit {unitname[0]} due to curve fitting failure.")
            continue

        plt.figure()
        plt.plot(x, y, 'bx')
        plt.plot(x1, piecewise_cubic_quad(x1, *p_best), 'r')
        plt.scatter(p_best[0], p_best[1], s=80, c='r')
        plt.xlabel('Load(MW)')
        plt.ylabel('DEF(ton/MW)')
        plt.title(unitname[0] + '_Piecewise-CQ')
        ensure_dir(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}')
        plt.savefig(f'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_fig/{f}/{unitname[0]}_Piecewise-CQ.png')
        plt.close()

        f_pcq1 = '%f x*x*x + %f x*x + %f x+ %f' % (p_best[2], p_best[3], p_best[4], p_best[1]-p_best[2]*p_best[0]*p_best[0]*p_best[0]-p_best[3]*p_best[0]*p_best[0]-p_best[4]*p_best[0])
        f_pcq2 = '%f x*x + %f x+%f' % (p_best[5], p_best[6], p_best[1]-p_best[5]*p_best[0]*p_best[0]-p_best[6]*p_best[0])
        print("Piecewise-Cubic-Quad Fitting For %s: %s and %s" % (str(unitname[0]), f_pcq1, f_pcq2))
        mse_pcq = mean_squared_error(y, piecewise_cubic_quad(xlist, *p_best))
        mape_pcq = mean_absolute_percentage_error(y, piecewise_cubic_quad(xlist, *p_best))

        # 比较所有模型的 MSE 和 MAPE，找出最优模型
        MSE_all = [mse_linear, mse_quad, mse_cubic, mse_plinear, mse_pql, mse_pcq]
        mse_min = min(MSE_all)
        mse_min_index = MSE_all.index(mse_min)

        MAPE_all = [mape_linear, mape_quad, mape_cubic, mape_plinear, mape_pql, mape_pcq]
        mape_min = min(MAPE_all)
        mape_min_index = MAPE_all.index(mape_min)

        Model_all = ['Linear', 'Quadratic', 'Cubic', 'Piecewise-Linear', 'Piecewise-Quad-Linear', 'Piecewise-Cubic-Quad']
        model_mse = Model_all[mse_min_index]
        model_mape = Model_all[mape_min_index]
        print("For %s,\n%s model has minimum MSE = %f, %s model has minimum MAPE = %f\n"
              % (unitname[0], model_mse, mse_min, model_mape, mape_min))

        # 将拟合结果保存到 FIT 和 MSE 列表中
        FIT.append([unitname[0], Aef, Aef_EPA, f_linear, f_quad, f_cubic, (f_plinear1, f_plinear2), (f_pql1, f_pql2), (f_pcq1, f_pcq2)])
        MSE.append([unitname[0], mse_linear, mse_quad, mse_cubic, mse_plinear, mse_pql, mse_pcq])

# 确保MSE列表包含数据
if MSE:
    MSE_sum = np.sum([i[1:] for i in MSE], axis=0)
    if MSE_sum.ndim > 0:
        MSE.append(['sum', *MSE_sum])
    MSE_df = pd.DataFrame(MSE, columns=['unit', 'Linear', 'Quadratic', 'Cubic', 'Piecewise-Linear', 'Piecewise-Quad-Linear', 'Piecewise-Cubic-Quad'])
    ensure_dir('/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_result')
    MSE_df.to_csv('/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_result/fitting_mse_emission.csv', encoding="utf_8_sig", index=False, header=True, mode='w')
else:
    print("No valid MSE data to save.")

if FIT:
    FIT_df = pd.DataFrame(FIT, columns=['unit', 'AEF', 'AEF_EPA', 'Linear', 'Quadratic', 'Cubic', 'Piecewise-Linear', 'Piecewise-Quad-Linear', 'Piecewise-Cubic-Quad'])
    ensure_dir('/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_result')
    FIT_df.to_csv('/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/Fitting_result/fitting_curve_emission.csv', encoding="utf_8_sig", index=False, header=True, mode='w')
else:
    print("No valid FIT data to save.")
