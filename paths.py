"""路徑：便攜版把設定、校正、模型都放在 EXE 旁邊。"""
import os
import sys


def base_dir():
    """使用者資料（config.json、calibration.json、models/、log.txt）的位置。"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_dir():
    """打包進 EXE 的唯讀檔案（web 介面）的位置。"""
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))


def data(*parts):
    return os.path.join(base_dir(), *parts)


def resource(*parts):
    return os.path.join(resource_dir(), *parts)
