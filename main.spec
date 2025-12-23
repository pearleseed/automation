# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Auto C-Peach
Windows-only build configuration

Build command:
    pyinstaller main.spec

Requirements:
    - PyInstaller >= 5.0
    - All dependencies from requirements.txt installed
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the project root directory
project_root = os.path.abspath(SPECPATH)

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[
        # OneOCR DLL files (Windows-only)
        ('.config/oneocr/oneocr.dll', '.config/oneocr'),
        ('.config/oneocr/onnxruntime.dll', '.config/oneocr'),
    ],
    datas=[
        # OneOCR model file
        ('.config/oneocr/oneocr.onemodel', '.config/oneocr'),
        # Data files
        ('data/festivals.json', 'data'),
        ('data/hopping.csv', 'data'),
    ],
    hiddenimports=[
        # Tkinter GUI (sometimes needs explicit import)
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.scrolledtext',
        
        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'numpy',
        
        # Airtest automation
        'airtest',
        'airtest.core',
        'airtest.core.api',
        'airtest.core.error',
        
        # OneOCR (custom module)
        'oneocr',
        
        # Project modules
        'core',
        'core.agent',
        'core.base',
        'core.config',
        'core.data',
        'core.detector',
        'core.oneocr_optimized',
        'core.utils',
        'automations',
        'automations.festivals',
        'automations.gachas',
        'automations.hopping',
        'gui',
        'gui.components',
        'gui.components.base_tab',
        'gui.components.quick_actions_panel',
        'gui.tabs',
        'gui.tabs.festival_tab',
        'gui.tabs.gacha_tab',
        'gui.tabs.hopping_tab',
        'gui.utils',
        'gui.utils.logging_utils',
        'gui.utils.thread_utils',
        'gui.utils.ui_utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'setuptools',
        'pip',
        'wheel',
        # Exclude YOLO/ultralytics/PyTorch (completely removed)
        'ultralytics',
        'torch',
        'torchvision',
        'torchaudio',
        'onnx',
        'onnxruntime-gpu',
        'tensorboard',
        'tensorflow',
        # Exclude web server modules (not needed for GUI)
        'uvicorn',
        'fastapi',
        'starlette',
        'httptools',
        'uvloop',
        'websockets',
        # Exclude test modules
        'unittest',
        'test',
        'tests',
        # Additional excludes for smaller build
        'email',
        'html',
        'http',
        'xml',
        'pydoc',
        'doctest',
        'argparse',
        'optparse',
        'getopt',
        'bdb',
        'pdb',
        'profile',
        'cProfile',
        'timeit',
        'trace',
        'lib2to3',
        'distutils',
        'ensurepip',
        'venv',
        'idlelib',
        'tkinter.tix',
        'turtle',
        'turtledemo',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# EXE configuration
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoCPeach',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available: icon='icon.ico'
)

# COLLECT - gather all files into dist folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoCPeach',
)
