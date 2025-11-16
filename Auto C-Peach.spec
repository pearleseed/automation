# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Auto C-Peach
Builds a standalone Windows application with all dependencies
Target Platform: Windows Only
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs
from pathlib import Path

# Block cipher (optional encryption - set to None for no encryption)
block_cipher = None

# Get the base path
base_path = Path('.').resolve()

# ==================== DATA FILES ====================
# Collect data files that need to be bundled

datas = []


# Collect minimal data files from dependencies (required for libraries to work)
# OneOCR data files
try:
    datas += collect_data_files('oneocr')
except Exception:
    pass

# Ultralytics data files (YOLO configs, etc.)
try:
    datas += collect_data_files('ultralytics')
except Exception:
    pass

# ==================== HIDDEN IMPORTS ====================
# Modules that PyInstaller might miss during automatic analysis

hiddenimports = [
    # Core Python modules
    'queue',
    'logging',
    'logging.handlers',
    
    # Tkinter components (GUI framework)
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.scrolledtext',
    
    # Computer Vision
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    
    # OCR Engine
    'oneocr',
    
    # Airtest automation framework
    'airtest',
    'airtest.core',
    'airtest.core.api',
    'airtest.core.error',
    'airtest.core.cv',
    'airtest.core.helper',
    'airtest.aircv',
    
    # YOLO / Ultralytics
    'ultralytics',
    'ultralytics.nn',
    'ultralytics.nn.modules',
    'ultralytics.engine',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.utils',
    
    # PyTorch (required by YOLO)
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.transforms',
    
    # Additional dependencies
    'yaml',
    'matplotlib',
    'scipy',
    'pandas',
    'requests',
    
    # Application modules
    'core',
    'core.agent',
    'core.base',
    'core.config',
    'core.data',
    'core.detector',
    'core.utils',
    'automations',
    'automations.festivals',
    'automations.gachas',
    'automations.hopping',
    'gui',
    'gui.tabs',
    'gui.tabs.festival_tab',
    'gui.tabs.gacha_tab',
    'gui.tabs.hopping_tab',
    'gui.components',
    'gui.components.base_tab',
    'gui.components.progress_panel',
    'gui.components.quick_actions_panel',
    'gui.utils',
    'gui.utils.logging_utils',
    'gui.utils.thread_utils',
    'gui.utils.ui_utils',
]

# Collect all submodules from key packages
try:
    hiddenimports += collect_submodules('ultralytics')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('torch')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('airtest')
except Exception:
    pass

# ==================== BINARIES ====================
# Collect dynamic libraries

binaries = []

# Collect dynamic libraries from packages
try:
    binaries += collect_dynamic_libs('cv2')
except Exception:
    pass

try:
    binaries += collect_dynamic_libs('torch')
except Exception:
    pass

try:
    binaries += collect_dynamic_libs('oneocr')
except Exception:
    pass

# ==================== ANALYSIS ====================
# Analyze the main script and gather all dependencies

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib.tests',
        'numpy.tests',
        'PIL.tests',
        'test',
        'tests',
        'unittest',
        'setuptools',
        'distutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ==================== PYZ (Python Archive) ====================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ==================== EXE (Executable) ====================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Auto C-Peach',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed mode (-w flag) - no console window
    disable_windowed_traceback=False,
    target_arch=None,
    icon=None,  # Add your Windows icon here: icon='icon.ico'
)

# ==================== COLLECT (Bundle Everything) ====================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Auto C-Peach',
)

# Build complete - executable will be in: dist/Auto C-Peach/Auto C-Peach.exe

