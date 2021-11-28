import sys
import os
from cx_Freeze import setup, Executable

# ADD FILES
files = ['icon.ico','gui/','app/','settings.json']

# TARGET
target = Executable(
    script="main.py",
    base="Win32GUI",
    icon="icon.ico"
)

# SETUP CX FREEZE
setup(
    name = "SigSYS",
    version = "1.0",
    description = "电磁信号识别验证系统",
    author = "Junjie Ren",
    options = {'build_exe' : {'include_files' : files}},
    executables = [target]
    
)