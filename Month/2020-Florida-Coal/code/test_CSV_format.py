"""
Created on 2024/06/29

@author: Simon
"""
import chardet

# 文件路径
file_path = '/Users/oo/Desktop/carbon_emission_model/CEMS_plant/hourly-emissions-2020-NY.csv'
##'/Users/oo/Desktop/carbon_emission_model/CEMS_plant_filter/CSV/P100/emissions-hourly-2020-ca_unit1B_P62.csv'
##'/Users/oo/Desktop/carbon_emission_model/CEMS_plant/emissions-hourly-2020-ca.csv'

# 打开文件并读取内容，以便检测编码
with open(file_path, 'rb') as file:
    # 读取文件的前几千字节，通常足以确定编码
    rawdata = file.read(5000)
    result = chardet.detect(rawdata)
    encoding = result['encoding']

# 输出检测到的编码格式
print("Detected encoding:", encoding)
