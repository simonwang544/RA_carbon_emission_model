# -*- coding: utf-8 -*-
"""
Created on 2024/07/15
地区: Texas & Indiana
燃烧类型为:Coal
时间为:2000/01/01-2024/04/30
作者: natur
3个筛选条件
以季度为单位
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
print(f"合并数据，共 {all_data.shape[0]} 行。")

# 尝试将日期转换为 datetime 类型
all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')

# 按时间排序
all_data.sort_values(by='Date', inplace=True)

# 过滤时间范围
start_time = datetime.strptime('20000101', '%Y%m%d')
end_time = datetime.strptime('20240430', '%Y%m%d')
filtered_data = all_data[(all_data['Date'] >= start_time) & (all_data['Date'] <= end_time)].copy()
print(f"过滤后的数据，共 {filtered_data.shape[0]} 行。")

# 生成唯一机组标识符
filtered_data['Unit_Identifier'] = (
    filtered_data['State'] + '_' +
    filtered_data['Facility Name'] + '_' + 
    filtered_data['Facility ID'].astype(str) + '_' + 
    filtered_data['Unit ID'].astype(str)
)

print(f"生成唯一机组标识符。")

# 记录生成文件的数量
generated_files = 0

# 按照前述步骤处理数据
for unit_identifier in filtered_data['Unit_Identifier'].drop_duplicates():
    df_unit = filtered_data[filtered_data['Unit_Identifier'] == unit_identifier].copy()  # 确保我们正在操作 DataFrame 的副本
    dedup = df_unit.drop_duplicates(subset=['Gross Load (MW)'])
    load_value = dedup['Gross Load (MW)'].dropna()

    print(f"处理机组 {unit_identifier}，有效负荷值数量：{len(load_value)}")

    if len(load_value) >= 30:
        load_max = int(max(load_value))       
        if len(load_value) >= 0.9 * load_max:
            DEF = df_unit['CO2 Mass (short tons)'] / df_unit['Gross Load (MW)']
            Gload = df_unit['Gross Load (MW)']

            print(f"机组 {unit_identifier} 的最大负荷值：{load_max}")

            # 删除无穷大和无穷小的 DEF 值，并保持 Gload 和 DEF 一致
            valid_indices = DEF.replace([np.inf, -np.inf], np.nan).dropna().index
            DEF = DEF.loc[valid_indices]
            Gload = Gload.loc[valid_indices]

            if len(DEF.drop_duplicates().dropna()) >= 0.9 * load_max:
                print(f"机组 {unit_identifier} 满足生成文件条件")
                # 插入 DEF 列到第 9 列（即 'J' 列）
                df_unit.insert(9, 'DEF(ton/MW)', np.nan)
                df_unit.loc[valid_indices, 'DEF(ton/MW)'] = DEF
                # 计算 AEF
                AEF = DEF.mean() if not DEF.empty else np.nan
                # 插入 AEF 列到第 10 列（即 'K' 列）
                df_unit.insert(10, 'AEF(ton/MW)', np.nan)
                df_unit.loc[valid_indices, 'AEF(ton/MW)'] = AEF
                
                state_name = df_unit['State'].iloc[0]
                facility_name = df_unit['Facility Name'].iloc[0]
                facility_id = df_unit['Facility ID'].iloc[0]
                unit_id = df_unit['Unit ID'].iloc[0]

                filename_base = f"{state_name}_{facility_name}_{facility_id}_{unit_id}_P{load_max}"
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
                df_unit.to_csv(filename_csv, encoding="utf_8_sig", index=False, header=True, mode='w', float_format='%.15g')

                # 增加生成文件数量计数
                generated_files += 1
            else:
                print(f"机组 {unit_identifier} 的 DEF 数据非零非重复的值较多")
        else:
            print(f"机组 {unit_identifier} 的出力变化范围比较小")
    else:
        print(f"机组 {unit_identifier} 的有效负荷值数量不足 30")

# 验证是否有文件生成
if generated_files > 0:
    print(f"生成了 {generated_files} 个CSV文件。")
else:
    print("没有生成任何CSV文件。")
