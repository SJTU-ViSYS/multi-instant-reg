'''
Date: 2021-05-15 17:05:11
LastEditors: ze bai
LastEditTime: 2021-05-15 17:17:09
FilePath: /classify_exp/utils/logger.py
'''
import time
 
# 格式化成2016-03-20 11:45:39形式 
class Logger:
    def __init__(self, path):
        self.path = path
        self.fw = open(self.path,'a')

    def info(self, text):
        self.fw.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {text}\n')
        self.fw.flush()

    def close(self):
        self.fw.close()