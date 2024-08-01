# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:47:10 2022

@author: natur
"""
# 1filter:按机组来分；gload区间完整度作图;筛选机组；先看2021一年的；
# 2处理缺失 P=0
# 3计算DEF 重复值
# 4作图, 对应机组类型

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 使用 os.path.join 构建路径
Folder_Path = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant')
SaveFile_Path100 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'CSV', 'P100')
SaveFIG_Path100 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'FIG', 'P100')
SaveFile_Path500 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'CSV', 'P500')
SaveFIG_Path500 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'FIG', 'P500')
SaveFile_Path1000 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'CSV', 'P1000')
SaveFIG_Path1000 = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'FIG', 'P1000')
SaveFile_PathBig = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'CSV', 'PBig')
SaveFIG_PathBig = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'FIG', 'PBig')
SaveFIG_NewPig = os.path.join(os.path.dirname(os.getcwd()), 'CEMS_plant_filter', 'NEW_PIG')

# 确认目录存在
os.makedirs(Folder_Path, exist_ok=True)
os.makedirs(SaveFile_Path100, exist_ok=True)
os.makedirs(SaveFIG_Path100, exist_ok=True)
os.makedirs(SaveFile_Path500, exist_ok=True)
os.makedirs(SaveFIG_Path500, exist_ok=True)
os.makedirs(SaveFile_Path1000, exist_ok=True)
os.makedirs(SaveFIG_Path1000, exist_ok=True)
os.makedirs(SaveFile_PathBig, exist_ok=True)
os.makedirs(SaveFIG_PathBig, exist_ok=True)
os.makedirs(SaveFIG_NewPig, exist_ok=True)

os.chdir(Folder_Path)
file_list = os.listdir(Folder_Path)

# 忽略 .DS_Store 文件
file_list = [f for f in file_list if not f.startswith('.')]

# 1
for i in range(0, len(file_list)):  # len(file_list)
    df = pd.read_csv(os.path.join(Folder_Path, file_list[i]), engine='c', encoding='latin1')
    Gload = df['Gross Load (MW)']
    plt.plot(Gload)
    name = file_list[i].split(".")
    plt.title(name[0])
    plt.savefig(os.path.join(SaveFIG_Path100, name[0] + '.svg'))
    plt.show()

# 自动筛选
for i in range(0, 1):  # 0, len(file_list)
    df0 = pd.read_csv(os.path.join(Folder_Path, file_list[i]), engine='c', encoding='latin1')
    start_time = datetime.strptime('20200101', '%Y%m%d')
    end_time = datetime.strptime('20210101', '%Y%m%d')
    df1 = df0[(start_time <= pd.to_datetime(df0['Date'])) & (pd.to_datetime(df0['Date']) < end_time)]
    for j in df1['Unit ID'].drop_duplicates():
        df = df1[df1['Unit ID'] == j]
        dedup = df.drop_duplicates(subset=['Gross Load (MW)'])
        load_value = dedup['Gross Load (MW)'].dropna()
        if len(load_value) >= 30:
            load_max = int(max(load_value))
            load_min = min(load_value)
            if len(load_value) >= 0.9 * load_max:
                # 计算 DEF
                DEF = df['CO2 Mass (short tons)'] / df['Gross Load (MW)']
                Gload = df['Gross Load (MW)']
                if len(DEF.drop_duplicates().dropna()) >= 0.9 * load_max:
                    plantname = file_list[i].split(".")
                    plt.figure()
                    plt.plot(Gload, DEF, 'o', markersize=1.5)
                    plt.xlabel('Load(MW)')
                    plt.ylabel('DEF(ton/MW)')
                    plt.title(plantname[0] + '_unit' + str(j) + '_P' + str(load_max))

                    df.insert(8, 'DEF(ton/MW)', DEF)

                    # 生成以时间为自变量的散点图
                    df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
                    df = df.sort_values('Datetime')
                    time_seq = (df['Datetime'] - df['Datetime'].min()).dt.total_seconds() / 3600

                    plt.figure()
                    plt.plot(time_seq, DEF, 'o', markersize=1.5)
                    plt.xlabel('Time (hours)')
                    plt.ylabel('DEF (ton/MW)')
                    plt.title(plantname[0] + '_unit' + str(j) + '_DEF_vs_Time')
                    plt.savefig(os.path.join(SaveFIG_NewPig, plantname[0] + '_unit' + str(j) + '_DEF_vs_Time.svg'))
                    plt.show()

                    if type(j) == str:
                        if not j.isalnum():
                            j = 'xxx' + ''.join(filter(str.isalnum, j))
                    if 0 < load_max <= 100:
                        filename = os.path.join(SaveFile_Path100, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.csv')
                        plt.savefig(os.path.join(SaveFIG_Path100, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.svg'))
                        plt.show()
                        df.to_csv(filename, encoding="utf_8_sig", index=False, header=True, mode='w')
                    elif 100 < load_max <= 500:
                        filename = os.path.join(SaveFile_Path500, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.csv')
                        plt.savefig(os.path.join(SaveFIG_Path500, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.svg'))
                        plt.show()
                        df.to_csv(filename, encoding="utf_8_sig", index=False, header=True, mode='w')
                    elif 500 < load_max <= 1000:
                        filename = os.path.join(SaveFile_Path1000, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.csv')
                        plt.savefig(os.path.join(SaveFIG_Path1000, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.svg'))
                        plt.show()
                        df.to_csv(filename, encoding="utf_8_sig", index=False, header=True, mode='w')
                    else:
                        filename = os.path.join(SaveFile_PathBig, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.csv')
                        plt.savefig(os.path.join(SaveFIG_PathBig, plantname[0] + '_unit' + str(j) + '_P' + str(load_max) + '.svg'))
                        plt.show()
                        df.to_csv(filename, encoding="utf_8_sig", index=False, header=True, mode='w')
