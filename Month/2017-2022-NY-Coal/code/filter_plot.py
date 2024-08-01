# -*- coding: utf-8 -*-
"""
Created on 2024/07/15
地区为New York
燃烧类型为:Coal
时间为:2017/01/01-2022/12/31

@author: natur
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

Folder_Path = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant'
SaveFile_Path100 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/P100'
SaveFIG_Path100 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/FIG/P100'
SaveFile_Path500 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/P500'
SaveFIG_Path500 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/FIG/P500'
SaveFile_Path1000 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/P1000'
SaveFIG_Path1000 = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/FIG/P1000'
SaveFile_PathBig = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/PBig'
SaveFIG_PathBig = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/FIG/PBig'

os.chdir(Folder_Path)
file_list = [f for f in os.listdir(Folder_Path) if f.endswith('.csv')]

# 1
for i in range(len(file_list)):
    df = pd.read_csv(os.path.join(Folder_Path, file_list[i]), encoding='latin1', low_memory=False)
    ##print(f"Processing file {file_list[i]}")  # 打印正在处理的文件名
    if 'Gross Load (MW)' in df.columns:
        Gload = df['Gross Load (MW)']
    else:
        print(f"Column 'Gross Load (MW)' not found in file {file_list[i]}")
        continue
    plt.plot(Gload)
    name = file_list[i].split(".")
    plt.title(name[0])
    plt.savefig(os.path.join(SaveFIG_PathBig, name[0] + '.svg'))
    plt.show()

# 自动筛选
start_time = datetime.strptime('20170101', '%Y%m%d')
end_time = datetime.strptime('20221231', '%Y%m%d')

for i in range(len(file_list)):
    df0 = pd.read_csv(os.path.join(Folder_Path, file_list[i]), encoding='latin1', low_memory=False)
    df1 = df0[(start_time <= pd.to_datetime(df0['Date'])) & (pd.to_datetime(df0['Date']) <= end_time)]
    
    for j in df1['Unit ID'].drop_duplicates():
        df = df1[df1['Unit ID'] == j].copy()  # 确保我们正在操作 DataFrame 的副本
        dedup = df.drop_duplicates(subset=['Gross Load (MW)'])
        load_value = dedup['Gross Load (MW)'].dropna()
        
        if len(load_value) >= 30:
            load_max = int(max(load_value))
            load_min = min(load_value)
            
            if len(load_value) >= 0.9 * load_max:
                DEF = df['CO2 Mass (short tons)'] / df['Gross Load (MW)']
                Gload = df['Gross Load (MW)']
                
                if len(DEF.drop_duplicates().dropna()) >= 0.9 * load_max:
                    # 插入 DEF 列到第 9 列（即 'J' 列）
                    df.insert(9, 'DEF(ton/MW)', DEF)
                    # 排除无效值并计算 AEF 列，跳过空值
                    valid_DEF = DEF.replace([np.inf, -np.inf], np.nan).dropna()
                    AEF = valid_DEF.mean() if not valid_DEF.empty else np.nan
                    # 插入 AEF 列到第 10 列（即 'K' 列）
                    df.insert(10, 'AEF(ton/MW)', AEF)
                    
                    plantname = file_list[i].split(".")
                    plt.figure()
                    plt.plot(Gload, DEF, 'o', markersize=1.5)
                    plt.xlabel('Load(MW)')
                    plt.ylabel('DEF(ton/MW)')
                    plt.title(f"{plantname[0]}_unit{j}_P{load_max}")
                    
                    if type(j) == str and not j.isalnum():
                        j = 'xxx' + ''.join(filter(str.isalnum, j))
                    
                    if 0 < load_max <= 100:
                        filename = os.path.join(SaveFile_Path100, f"{plantname[0]}_unit{j}_P{load_max}.csv")
                        plt.savefig(os.path.join(SaveFIG_Path100, f"{plantname[0]}_unit{j}_P{load_max}.svg"))
                    elif 100 < load_max <= 500:
                        filename = os.path.join(SaveFile_Path500, f"{plantname[0]}_unit{j}_P{load_max}.csv")
                        plt.savefig(os.path.join(SaveFIG_Path500, f"{plantname[0]}_unit{j}_P{load_max}.svg"))
                    elif 500 < load_max <= 1000:
                        filename = os.path.join(SaveFile_Path1000, f"{plantname[0]}_unit{j}_P{load_max}.csv")
                        plt.savefig(os.path.join(SaveFIG_Path1000, f"{plantname[0]}_unit{j}_P{load_max}.svg"))
                    else:
                        filename = os.path.join(SaveFile_PathBig, f"{plantname[0]}_unit{j}_P{load_max}.csv")
                        plt.savefig(os.path.join(SaveFIG_PathBig, f"{plantname[0]}_unit{j}_P{load_max}.svg"))
                    
                    plt.show()
                    df.to_csv(filename, encoding="utf_8_sig", index=False, header=True, mode='w')
