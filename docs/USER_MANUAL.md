# Auto C-Peach - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Quick Start Guide](#quick-start-guide)
4. [Application Overview](#application-overview)
5. [Core Features](#core-features)
6. [Detailed Usage Guides](#detailed-usage-guides)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

**Auto C-Peach** is a comprehensive game automation tool designed for automating repetitive tasks in DOAX VenusVacation. The application provides three main automation modules:

- **Festival Automation**: Automates festival battles with OCR verification
- **Gacha Automation**: Automates gacha pulls with result detection
- **Hopping Automation**: Automates world hopping with verification

### Key Features

- **GUI-Based Interface**: Easy-to-use graphical interface for all operations
- **OCR Technology**: Advanced OCR for text recognition and verification
- **Template Matching**: Image-based detection using template matching
- **YOLO Detection**: Optional AI-powered object detection (YOLO)
- **Structured Logging**: Comprehensive logging with file output
- **Resume Support**: Resume interrupted automation sessions
- **Progress Tracking**: Real-time progress monitoring
- **Error Handling**: Robust error handling with retry mechanisms

---

## System Requirements

### Operating System
- **Windows 10/11**

### Software Requirements
- **Python 3.8+** (Python 3.13 recommended)
- **DOAX VenusVacation** game running
- **Airtest Framework** (included in dependencies)

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 500MB for application + templates
- **GPU**: Optional (for YOLO acceleration with CUDA/MPS)

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- `opencv-python>=4.10.0`
- `numpy>=2.0.0,<3.0.0`
- `ultralytics>=8.0.0` (for YOLO)
- `oneocr>=1.0.0` (for OCR)
- `airtest>=1.3.0` (for device control)

---
## Quick Start Guide

### 1. Launch the Application

```bash
python main.py
```

### 2. Connect Your Device

1. Ensure **DOAX VenusVacation** is running
2. Click **"Connect Device"** in the main window
3. Wait for "Device Connected" status

### 3. Choose Your Automation

- **Festival Tab**: For automating festival battles
- **Gacha Tab**: For automating gacha pulls
- **Hopping Tab**: For automating world hopping

### 4. Configure and Run

1. Select your data file (CSV/JSON) or configure settings
2. Click **"Start"** to begin automation
3. Monitor progress in the log viewer

---

## Application Overview

### Main Window Structure

```
┌─────────────────────────────────────────────────┐
│  Auto C-Peach                    [Connect] [Refresh] │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Festival] [Gacha] [Hopping] [Settings]      │
│                                                 │
│  ┌─────────────────┐  ┌──────────────┐        │
│  │ Configuration   │  │ Progress     │        │
│  │ Panel           │  │ Panel        │        │
│  │                 │  │              │        │
│  └─────────────────┘  └──────────────┘        │
│                                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ Log Viewer                              │  │
│  │                                         │  │
│  └─────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Agent Module** (`core/agent.py`)
- Device connection and control
- Screen capture (snapshots)
- OCR processing
- Touch/screen interaction

#### 2. **Base Automation** (`core/base.py`)
- Common automation functionality
- Template matching
- ROI (Region of Interest) processing
- OCR text extraction
- Step execution with retry logic

#### 3. **Detector Module** (`core/detector.py`)
- **YOLODetector**: AI-based object detection
- **TemplateMatcher**: Image template matching
- **OCRTextProcessor**: Advanced text processing

#### 4. **Data Module** (`core/data.py`)
- CSV/JSON file loading
- Result writing with resume support
- Data validation

#### 5. **Configuration** (`core/config.py`)
- Centralized configuration management
- ROI definitions
- Automation settings

---

## Core Features

### 1. Device Connection

The application connects to DOAX VenusVacation using Airtest framework.

**Connection Process:**
1. Click **"Connect Device"** button
2. Application searches for game window
3. Connection verified with test screenshot
4. Status updates to "Device Connected"

**Troubleshooting:**
- Ensure game is running
- Check window title matches pattern: `DOAX VenusVacation.*`
- Try clicking **"Refresh"** button

### 2. OCR (Optical Character Recognition)

The application uses **OneOCR** for text recognition.

**Features:**
- Full-screen OCR
- Region-specific OCR (ROI)
- Text extraction with coordinates
- Fuzzy text matching

**Usage:**
- OCR runs automatically during automation
- Can be tested via **"OCR Test"** quick action

### 3. Template Matching

Template matching uses image comparison to find UI elements.

**Template Requirements:**
- PNG format recommended
- Clear, high-contrast images
- Minimal background noise
- Consistent size/scale

**Template Locations:**
- Base templates: `./templates/`
- Banner templates: `./templates/banners/`

### 4. YOLO Detection (Optional)

YOLO (You Only Look Once) provides AI-powered object detection.

**Requirements:**
- YOLO model file (`yolo11n.pt` or custom)
- GPU recommended (but not required)

**Configuration:**
- Enable in Festival automation settings
- Configure confidence threshold
- Select device (CPU/CUDA/MPS)

---

## Detailed Usage Guides

### Festival Automation

#### Overview

Festival Automation automates festival battles with OCR verification. It processes stages from a CSV/JSON file and verifies results.

#### Workflow

```
1. Navigate to Festival → Event
2. Take snapshot (before)
3. Find and select stage (OCR)
4. Find and select rank (OCR)
5. Take snapshot (after)
6. Verify pre-battle data (OCR + optional detector)
7. Start battle
8. Skip animation
9. View results
10. Take snapshot (result)
11. Verify post-battle data (OCR + optional detector)
12. Close result dialogs
13. Repeat for next stage
```

#### Step-by-Step Guide

**1. Prepare Data File**

Create a CSV or JSON file with festival data:

**CSV Format:**
```csv
フェス名,フェスランク,推奨ランク,勝利点数,Sランクボーダー,初回クリア報酬,Sランク報酬
イベント名,E,E,1000,500,アイテム1,アイテム2
```

**JSON Format:**
```json
[
  {
    "フェス名": "イベント名",
    "フェスランク": "E",
    "推奨ランク": "E",
    "勝利点数": "1000",
    "Sランクボーダー": "500"
  }
]
```

**2. Prepare Templates**

Required templates in `./templates/`:
- `tpl_festival.png` - Festival button
- `tpl_event.png` - Event button
- `tpl_challenge.png` - Challenge button
- `tpl_ok.png` - OK/Confirm button
- `tpl_allskip.png` - Skip animation button
- `tpl_result.png` - Result button

**3. Configure Settings**

In **Festival Tab**:
- **Data File**: Select your CSV/JSON file
- **Templates Folder**: Path to templates (default: `./templates`)
- **Output File**: Optional (auto-generated if empty)
- **Snapshots Folder**: Where screenshots are saved
- **Results Folder**: Where results CSV is saved

**4. Run Automation**

1. Click **"Start Festival"**
2. Monitor progress in log viewer
3. Check results in output CSV file

#### ROI Configuration

Festival automation uses predefined ROI regions for data extraction:

- **フェス名**: Festival name area
- **フェスランク**: Festival rank area
- **勝利点数**: Victory points area
- **推奨ランク**: Recommended rank area
- **Sランクボーダー**: S-rank border points
- **初回クリア報酬**: First clear reward
- **Sランク報酬**: S-rank reward
- **獲得ザックマネー**: Earned money
- **獲得アイテム**: Earned items
- **獲得EXP-Ace**: EXP for Ace units
- **獲得EXP-NonAce**: EXP for non-Ace units
- **エース**: Venus memory (Ace)
- **非エース**: Venus memory (non-Ace)

ROI coordinates are defined in `core/config.py` and can be adjusted if needed.

#### Verification Process

**Pre-Battle Verification:**
- Scans ROIs: `勝利点数`, `推奨ランク`, `Sランクボーダー`, `初回クリア報酬`, `Sランク報酬`
- Compares extracted data with expected values from CSV
- Retries if verification fails

**Post-Battle Verification:**
- Scans ROIs: `獲得ザックマネー`, `獲得アイテム`, `獲得EXP-Ace`, `獲得EXP-NonAce`, `エース`, `非エース`
- Compares extracted data with expected values
- Records OK/NG status

#### Output Files

**Screenshots:**
- Location: `result/festival/snapshots/rank_E_stage_1/`
- Files:
  - `01_before_touch.png` - Before selecting stage
  - `02_after_touch.png` - After selecting stage
  - `03_result.png` - Battle result screen

**Results CSV:**
- Location: `result/festival/results/results_YYYYMMDD_HHMMSS.csv`
- Columns: All input columns + `timestamp`, `result` (OK/NG), `error_message`

**Log Files:**
- Location: `result/festival/results/logs/festival_YYYYMMDD_HHMMSS.log`
- Contains detailed execution logs

#### Resume Support

If automation is interrupted:
1. Results are saved incrementally
2. Re-run with same output file
3. Completed stages are automatically skipped
4. Log shows: `✓ Stage X already completed, skipping...`

---

### Gacha Automation

#### Overview

Gacha Automation automates gacha pulls with result detection. It can pull multiple banners and detect SSR/SR + Swimsuit combinations.

#### Workflow

```
1. Find banner (scroll if needed)
2. Touch banner
3. Select pull type (single/multi)
4. Snapshot before pull
5. Confirm pull
6. Skip animation
7. Snapshot after pull
8. Check for SSR/SR + Swimsuit
9. Special snapshot if both found
10. Close result
11. Repeat for next pull
```

#### Step-by-Step Guide

**1. Prepare Banner Templates**

For each gacha banner, create a folder in `templates/banners/`:

```
templates/banners/summer_gacha/
├── banner.png              # REQUIRED: Banner image
├── swimsuit_red.png        # Swimsuit template 1
├── swimsuit_blue.png      # Swimsuit template 2
└── swimsuit_white.png    # Swimsuit template 3
```

**Requirements:**
- `banner.png` (or `.jpg`) is **REQUIRED**
- At least one swimsuit template is required
- Banner image should be clear and recognizable

**2. Prepare Base Templates**

Required templates in `./templates/`:
- `tpl_ssr.png` - SSR rarity icon
- `tpl_sr.png` - SR rarity icon
- `tpl_ok.png` - OK button
- `tpl_allskip.png` - Skip animation button
- `tpl_single_pull.png` - Single pull button
- `tpl_multi_pull.png` - Multi pull button
- `tpl_button_down.png` - Scroll down button

**3. Configure Gacha**

In **Gacha Tab**:

1. **Templates Folder**: Select templates folder (default: `./templates`)

2. **Select Banners**: 
   - UI displays all banners from `templates/banners/`
   - Click **"Add"** on desired banners
   - Icon ✓ = valid (has banner + swimsuit)
   - Icon ? = incomplete

3. **Configure Pull Settings**:
   - **Rarity**: SSR or SR (target rarity to detect)
   - **Pulls**: Number of pulls per banner
   - **Type**: Single or Multi (10x)

4. **Edit Selected Gachas** (optional):
   - Select gacha in list
   - Click **"Edit"** to modify pulls/type/rarity

**4. Run Automation**

1. Click **"Start Gacha"**
2. Automation processes each banner sequentially
3. For each pull:
   - Finds banner (scrolls if needed)
   - Performs pull
   - Checks for target rarity + swimsuit
   - Saves special snapshot if both found

#### Result Detection

**Detection Process:**
1. After pull, checks for rarity template (`tpl_ssr.png` or `tpl_sr.png`)
2. Checks for swimsuit templates in banner folder
3. If **both** found → saves `_SPECIAL.png` snapshot
4. Logs detection results

**Special Snapshots:**
- File naming: `{pull_idx:02d}_SPECIAL.png`
- Only created when both rarity and swimsuit detected
- Useful for tracking rare pulls

#### Output Files

**Screenshots:**
- Location: `result/gacha/snapshots/{idx:02d}_{banner_name}_{timestamp}/`
- Files:
  - `{pull_idx:02d}_before.png` - Before pull
  - `{pull_idx:02d}_after.png` - After pull
  - `{pull_idx:02d}_SPECIAL.png` - Special match (if found)

**Results CSV:**
- Location: `result/gacha/results/gacha_YYYYMMDD_HHMMSS.csv`
- Contains pull results and statistics

**Log Files:**
- Location: `result/gacha/results/logs/gacha_YYYYMMDD_HHMMSS.log`

#### Tips

- **Banner Visibility**: Ensure banner is visible on screen (or enable auto-scroll)
- **Template Quality**: Use high-quality screenshots for templates
- **Swimsuit Detection**: Multiple swimsuit templates increase detection accuracy
- **Pull Timing**: Adjust `wait_after_pull` if pulls are too fast/slow

---

### Hopping Automation

#### Overview

Hopping Automation automates world hopping with OCR verification. It hops between worlds and verifies successful transitions.

#### Workflow

```
1. Check current world (OCR)
2. Open world map
3. Touch hop button
4. Confirm hop
5. Wait for loading
6. Check new world (OCR)
7. Verify hop success (world changed)
8. Repeat for specified number of hops
```

#### Step-by-Step Guide

**1. Prepare Templates**

Required templates in `./templates/`:
- `tpl_world_map.png` - World map button
- `tpl_hop_button.png` - Hop button
- `tpl_confirm_hop.png` - Confirm hop button

**2. Configure Settings**

In **Hopping Tab**:

- **Number of Hops**: How many hops to perform (1-50)
- **Loading Wait**: Time to wait for world transition (seconds)
- **Default Pick**: Hopping Roulette option (1-6)

**3. Run Automation**

1. Click **"Start Hopping"**
2. Automation performs hops sequentially
3. Verifies each hop success
4. Displays results summary

#### Verification Process

**Hop Verification:**
1. Captures world name before hop (OCR)
2. Performs hop
3. Waits for loading
4. Captures world name after hop (OCR)
5. Compares names (normalized)
6. Verifies world changed

**Verification Logic:**
- Normalizes world names (removes spaces, converts case)
- Compares normalized names
- Checks similarity to avoid OCR false positives
- Logs verification results

#### Output Files

**Screenshots:**
- Location: `result/hopping/snapshots/hopping_session_{timestamp}/`
- Files:
  - `{hop_idx:02d}_before.png` - Before hop
  - `{hop_idx:02d}_after.png` - After hop

**Results CSV:**
- Location: `result/hopping/results/hopping_results_YYYYMMDD_HHMMSS.csv`
- Contains: `hop_number`, `world_before`, `world_after`, `success`, `timestamp`

**Log Files:**
- Location: `result/hopping/results/logs/hopping_YYYYMMDD_HHMMSS.log`

#### Batch Mode

Hopping supports batch processing from CSV/JSON file:

**CSV Format:**
```csv
session_id,num_hops,loading_wait,cooldown_wait
1,5,5.0,3.0
2,10,5.0,3.0
```

Each row represents a hopping session with specified number of hops.

---

## Configuration

### Application Settings

Access via **Settings Tab**:

**General Settings:**
- **Log Level**: DEBUG, INFO, WARNING, ERROR
- Controls verbosity of logging

**Performance Settings:**
- **Max Log Lines**: Maximum lines in log viewer (default: 1000)
- **Log Poll Interval**: Update frequency in milliseconds (default: 200ms)

### Automation Configuration

Each automation module has configurable parameters:

#### Festival Configuration

```python
{
    'templates_path': './templates',
    'snapshot_dir': './result/festival/snapshots',
    'results_dir': './result/festival/results',
    'wait_after_touch': 1.0,  # seconds
    'max_step_retries': 5,
    'retry_delay': 1.0,  # seconds
    'fuzzy_matching': {
        'enabled': True,
        'threshold': 0.7  # 0.0-1.0
    },
    'use_detector': True,  # Enable YOLO/Template detector
    'detector_type': 'template',  # 'yolo', 'template', 'auto'
    'yolo_config': {
        'model_path': 'yolo11n.pt',
        'confidence': 0.25,
        'device': 'cpu'  # 'cpu', 'cuda', 'mps', 'auto'
    }
}
```

#### Gacha Configuration

```python
{
    'templates_path': './templates',
    'snapshot_dir': './result/gacha/snapshots',
    'results_dir': './result/gacha/results',
    'wait_after_touch': 1.0,
    'wait_after_pull': 2.0,
    'max_pulls': 10,
    'pull_type': 'single',  # 'single' or 'multi'
    'max_scroll_attempts': 10
}
```

#### Hopping Configuration

```python
{
    'templates_path': './templates',
    'snapshot_dir': './result/hopping/snapshots',
    'results_dir': './result/hopping/results',
    'wait_after_touch': 1.0,
    'loading_wait': 5.0,  # seconds
    'cooldown_wait': 3.0,  # seconds
    'max_hops': 10,
    'retry_on_fail': True,
    'max_retries': 3
}
```

### ROI Configuration

ROI (Region of Interest) coordinates are defined in `core/config.py`.

**Format:**
```python
{
    "ROI_NAME": {
        "coords": [x1, x2, y1, y2],  # Pixel coordinates
        "description": "Description"
    }
}
```

**Adjusting ROI:**
1. Take screenshot of game screen
2. Identify region coordinates
3. Update `core/config.py`
4. Restart application

### Template Configuration

**Template Matching Settings:**
- **Threshold**: Matching confidence (0.0-1.0)
  - Higher = stricter matching
  - Default: 0.85
- **Method**: Matching algorithm
  - `TM_CCOEFF_NORMED` (default)
  - `TM_CCORR_NORMED`
  - `TM_SQDIFF_NORMED`

---

## Troubleshooting

### Common Issues

#### 1. Device Connection Failed

**Symptoms:**
- "Connection Failed" message
- Device status shows "Not Connected"

**Solutions:**
- Ensure DOAX VenusVacation is running
- Check window title matches pattern
- Try clicking **"Refresh"**
- Restart game and application
- Check Airtest compatibility

#### 2. Template Not Found

**Symptoms:**
- Log shows: `Template not found: tpl_xxx.png`
- Automation fails at specific step

**Solutions:**
- Verify template file exists in `./templates/`
- Check file name matches exactly (case-sensitive)
- Ensure file format is PNG/JPG
- Verify template image is clear and recognizable

#### 3. OCR Recognition Failed

**Symptoms:**
- Verification fails
- Incorrect text extracted

**Solutions:**
- Check ROI coordinates are correct
- Ensure text is visible and clear
- Adjust fuzzy matching threshold
- Verify game resolution matches expected
- Try adjusting wait times

#### 4. Automation Stuck/Freezing

**Symptoms:**
- Progress stops
- No log updates
- Application unresponsive

**Solutions:**
- Click **"Stop"** button
- Check log file for errors
- Verify device is still connected
- Restart application
- Check for game updates

#### 5. False Positive Detections

**Symptoms:**
- Wrong items detected
- Incorrect verification results

**Solutions:**
- Adjust detector confidence threshold
- Improve template quality
- Update ROI coordinates
- Enable/disable fuzzy matching
- Review screenshots for issues

### Debug Mode

Enable debug logging:

1. Go to **Settings Tab**
2. Set **Log Level** to **DEBUG**
3. Run automation
4. Review detailed logs

### Log Analysis

**Log File Locations:**
- Festival: `result/festival/results/logs/`
- Gacha: `result/gacha/results/logs/`
- Hopping: `result/hopping/results/logs/`

**Log Structure:**
```
================================================================================
 AUTOMATION START
================================================================================
[STEP  1] Step Name - START
[STEP  1] ✓ Step Name - SUCCESS
[STEP  2] Step Name - RETRY 1/5
...
================================================================================
 AUTOMATION END
================================================================================
```

### Getting Help

1. **Check Logs**: Review log files for error messages
2. **Screenshots**: Check snapshot folders for visual debugging
3. **Configuration**: Verify settings are correct
4. **Templates**: Ensure templates are up-to-date

---

## Best Practices

### Template Management

1. **Use High-Quality Screenshots**
   - Clear, high-contrast images
   - Minimal background noise
   - Consistent size/scale

2. **Organize Templates**
   - Group related templates in folders
   - Use descriptive names
   - Keep templates updated

3. **Test Templates**
   - Use "OCR Test" quick action
   - Verify template matching works
   - Update if game UI changes

### Data File Management

1. **CSV Format**
   - Use UTF-8 encoding
   - Include all required columns
   - Validate data before running

2. **JSON Format**
   - Valid JSON syntax
   - Array of objects
   - Consistent field names

3. **Backup Data**
   - Keep original data files
   - Version control recommended
   - Regular backups

### Automation Execution

1. **Monitor Progress**
   - Watch log viewer
   - Check progress panel
   - Review screenshots periodically

2. **Error Handling**
   - Don't interrupt automation unnecessarily
   - Let retry mechanism work
   - Review errors after completion

3. **Resource Management**
   - Close unnecessary applications
   - Ensure sufficient disk space
   - Monitor system resources

### Performance Optimization

1. **Log Settings**
   - Reduce max log lines for better performance
   - Increase poll interval if UI lags
   - Use INFO level for normal operation

2. **Template Optimization**
   - Use smallest necessary template size
   - Remove unnecessary templates
   - Optimize template matching threshold

3. **YOLO Configuration**
   - Use CPU if GPU unavailable
   - Adjust confidence threshold
   - Consider smaller model (yolo11n vs yolo11x)

---

## Advanced Topics

### Custom ROI Configuration

To add custom ROI regions:

1. Open `core/config.py`
2. Add ROI definition:
```python
CUSTOM_ROI_CONFIG = {
    "my_roi": {
        "coords": [x1, x2, y1, y2],
        "description": "My custom ROI"
    }
}
```
3. Use in automation:
```python
text = self.ocr_roi("my_roi")
```

### Custom Detector Integration

To integrate custom detector:

1. Create detector class inheriting from base
2. Implement `detect()` method
3. Return `List[DetectionResult]`
4. Configure in automation config

### Batch Processing

For processing multiple files:

1. Create script:
```python
from automations.festivals import FestivalAutomation
from core.agent import Agent

agent = Agent()
config = {...}

for file in files:
    automation = FestivalAutomation(agent, config)
    automation.run(file)
```

### API Usage

Automation classes can be used programmatically:

```python
from automations.festivals import FestivalAutomation
from core.agent import Agent
from core.config import get_festival_config

# Initialize
agent = Agent()
config = get_festival_config()

# Create automation
automation = FestivalAutomation(agent, config)

# Run
success = automation.run("data/festivals.csv")
```

### Logging Customization

Customize logging:

```python
from core.utils import StructuredLogger

logger = StructuredLogger(
    name="CustomLogger",
    log_file="custom.log"
)

logger.automation_start("CUSTOM AUTOMATION", {...})
```

---

## Appendix

### File Structure

```
automation/
├── main.py                 # Main application entry
├── requirements.txt        # Python dependencies
├── core/                   # Core modules
│   ├── agent.py           # Device control
│   ├── base.py            # Base automation
│   ├── config.py          # Configuration
│   ├── data.py            # Data handling
│   ├── detector.py        # Detection modules
│   └── utils.py           # Utilities
├── automations/            # Automation modules
│   ├── festivals.py       # Festival automation
│   ├── gachas.py          # Gacha automation
│   └── hopping.py         # Hopping automation
├── gui/                    # GUI components
│   ├── tabs/              # Tab implementations
│   ├── components/        # UI components
│   └── utils/             # UI utilities
├── templates/              # Template images
│   └── banners/           # Banner templates
├── data/                   # Data files
├── result/                 # Output files
│   ├── festival/
│   ├── gacha/
│   └── hopping/
└── docs/                   # Documentation
```

### ROI Coordinate System

Coordinates use pixel values:
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (increases)
- **Y-axis**: Top to bottom (increases)
- **Format**: `[x1, x2, y1, y2]`

Example: `[100, 500, 200, 400]`
- X: 100 to 500 (width: 400px)
- Y: 200 to 400 (height: 200px)

### Template Matching Methods

1. **TM_CCOEFF_NORMED** (Default)
   - Best for general use
   - Normalized correlation coefficient

2. **TM_CCORR_NORMED**
   - Cross-correlation
   - Good for similar images

3. **TM_SQDIFF_NORMED**
   - Squared difference
   - Lower values = better match

### YOLO Models

Available models:
- `yolo11n.pt` - Nano (smallest, fastest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (most accurate)

### Keyboard Shortcuts

- **Ctrl+C**: Copy selected text (in log viewer)
- **Ctrl+F**: Search (in data preview)
- **ESC**: Close dialogs

---

## Version History

- **v1.0** - Initial release
  - Festival automation
  - Gacha automation
  - Hopping automation
  - GUI interface
  - OCR support
  - Template matching
  - YOLO detection (optional)


