# -*- coding: utf-8 -*-
"""
Created on 2024/07/15
地区为 Florida
燃烧类型为:Coal
时间为:2017/01/01-2024/04/30
作者: natur
2个筛选条件
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

# 确保保存路径存在
os.makedirs(SaveFile_Path100, exist_ok=True)
os.makedirs(SaveFIG_Path100, exist_ok=True)
os.makedirs(SaveFile_Path500, exist_ok=True)
os.makedirs(SaveFIG_Path500, exist_ok=True)
os.makedirs(SaveFile_Path1000, exist_ok=True)
os.makedirs(SaveFIG_Path1000, exist_ok=True)
os.makedirs(SaveFile_PathBig, exist_ok=True)
os.makedirs(SaveFIG_PathBig, exist_ok=True)

os.chdir(Folder_Path)
file_list = [f for f in os.listdir(Folder_Path) if f.endswith('.csv')]

# 合并所有文件的数据
df_list = []
for file in file_list:
    df = pd.read_csv(os.path.join(Folder_Path, file), encoding='latin1', low_memory=False)
    df_list.append(df)

# 将所有数据合并到一个DataFrame中
all_data = pd.concat(df_list, ignore_index=True)

# 尝试将日期转换为 datetime 类型
all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')

# 按时间排序
all_data.sort_values(by='Date', inplace=True)

# 过滤时间范围
start_time = datetime.strptime('20170101', '%Y%m%d')
end_time = datetime.strptime('20221231', '%Y%m%d')
filtered_data = all_data[(all_data['Date'] >= start_time) & (all_data['Date'] <= end_time)].copy()

# 生成唯一机组标识符
filtered_data['Unit_Identifier'] = (
    filtered_data['Facility Name'] + '_' + 
    filtered_data['Facility ID'].astype(str) + '_' + 
    filtered_data['Unit ID'].astype(str)
)

# 记录生成文件的数量
generated_files = 0

# 按照前述步骤处理数据
for unit_identifier in filtered_data['Unit_Identifier'].drop_duplicates():
    df_unit = filtered_data[filtered_data['Unit_Identifier'] == unit_identifier].copy()  # 确保我们正在操作 DataFrame 的副本
    dedup = df_unit.drop_duplicates(subset=['Gross Load (MW)'])
    load_value = dedup['Gross Load (MW)'].dropna()

    if len(load_value) >= 30:
        load_max = int(max(load_value))       
        DEF = df_unit['CO2 Mass (short tons)'] / df_unit['Gross Load (MW)']
        Gload = df_unit['Gross Load (MW)']

        if len(DEF.drop_duplicates().dropna()) >= 0.9 * load_max:
            # 插入 DEF 列到第 9 列（即 'J' 列）
            df_unit.insert(9, 'DEF(ton/MW)', DEF)
            # 排除无效值并计算 AEF 列，跳过空值
            valid_DEF = DEF.replace([np.inf, -np.inf], np.nan).dropna()
            AEF = valid_DEF.mean() if not valid_DEF.empty else np.nan
            # 插入 AEF 列到第 10 列（即 'K' 列）
            df_unit.insert(10, 'AEF(ton/MW)', AEF)

            facility_name = df_unit['Facility Name'].iloc[0]
            facility_id = df_unit['Facility ID'].iloc[0]
            unit_id = df_unit['Unit ID'].iloc[0]

            filename_base = f"{facility_name}_{facility_id}_{unit_id}_P{load_max}"
            plt.figure()
            plt.plot(Gload, DEF, 'o', markersize=1.5)
            plt.xlabel('Load(MW)')
            plt.ylabel('DEF(ton/MW)')
            plt.title(filename_base)

            if 0 < load_max <= 100:
                filename_csv = os.path.join(SaveFile_Path100, filename_base + '.csv')
                filename_svg = os.path.join(SaveFIG_Path100, filename_base + '.svg')
            elif 100 < load_max <= 500:
                filename_csv = os.path.join(SaveFile_Path500, filename_base + '.csv')
                filename_svg = os.path.join(SaveFIG_Path500, filename_base + '.svg')
            elif 500 < load_max <= 1000:
                filename_csv = os.path.join(SaveFile_Path1000, filename_base + '.csv')
                filename_svg = os.path.join(SaveFIG_Path1000, filename_base + '.svg')
            else:
                filename_csv = os.path.join(SaveFile_PathBig, filename_base + '.csv')
                filename_svg = os.path.join(SaveFIG_PathBig, filename_base + '.svg')

            plt.savefig(filename_svg)
            plt.close()
            df_unit.to_csv(filename_csv, encoding="utf_8_sig", index=False, header=True, mode='w')

            # 增加生成文件数量计数
            generated_files += 1

# 验证是否有文件生成
if generated_files > 0:
    print(f"生成了 {generated_files} 个CSV文件。")
else:
    print("没有生成任何CSV文件。")
