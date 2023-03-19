# -*- coding:utf-8 -*-
import os
import logging as lg

class Logger:
    def __init__(self,fPath:str,fName:str,delete=True):
        self.logPath=os.path.join(fPath,fName)
        if not os.path.exists(self.logPath) or delete:
            open(self.logPath,'w').close()
        self.log=lg.getLogger(fName)
        #初始化设置
        self.log.setLevel(lg.DEBUG)
        fmt=lg.Formatter("[%(asctime)s] - %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")

        self.fh=lg.FileHandler(self.logPath,encoding='utf-8')
        self.fh.setFormatter(fmt)
        self.log.addHandler(self.fh)

        sh=lg.StreamHandler()
        sh.setFormatter(fmt)
        self.log.addHandler(sh)

        self.fh.close()

    def readLog(self)->str:
        f=open(self.logPath,'rb')
        data=f.readlines()
        f.close()
        return ''.join([i.decode('utf-8') for i in data])

    def __del__(self):
        lg.shutdown()
        if os.path.exists(self.logPath):
            while True:
                try:
                    os.remove(self.logPath)
                except PermissionError:
                    continue
                break

# if __name__=='__main__':
#     log=initLog('logs','log.log')
#     log.info('2222')