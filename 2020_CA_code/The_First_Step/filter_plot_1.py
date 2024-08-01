# -*- coding: utf-8 -*-
"""
Created on 2024/06/29

@author: Simon
"""
#1filter:按机组来分；gload区间完整度作图;筛选机组；先看2021一年的；
#2处理缺失 P=0
#3计算DEF 重复值
#4作图，对应机组类型

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

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
    ##print(f"Columns in file {file_list[i]}: {df.columns}")  # 打印列名
    ##print(df.dtypes)  # 打印每列的数据类型
    # 使用正确的列名
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

# 自动删选
for i in range(1):
    df0 = pd.read_csv(os.path.join(Folder_Path, file_list[i]), encoding='latin1', low_memory=False)
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
                DEF = df['CO2 Mass (short tons)'] / df['Gross Load (MW)']
                Gload = df['Gross Load (MW)']
                
                if len(DEF.drop_duplicates().dropna()) >= 0.9 * load_max:
                    plantname = file_list[i].split(".")
                    plt.figure()
                    plt.plot(Gload, DEF, 'o', markersize=1.5)
                    plt.xlabel('Load(MW)')
                    plt.ylabel('DEF(ton/MW)')
                    plt.title(f"{plantname[0]}_unit{j}_P{load_max}")
                    
                    df.insert(8, 'DEF(ton/MW)', DEF)
                    
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
