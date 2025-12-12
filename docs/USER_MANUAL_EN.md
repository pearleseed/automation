# Auto C-Peach User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started & Connection](#getting-started--connection)
3. [Main Interface](#main-interface)
4. [Festival Automation](#festival-automation-tab)
5. [Gacha Automation](#gacha-automation-tab)
6. [Hopping Automation](#hopping-automation-tab)
7. [Settings](#settings-tab)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Results & Screenshots](#results--screenshots)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## Introduction

Auto C-Peach is an automation tool for DOAX Venus Vacation with 3 main functions:

| Feature | Description |
|---------|-------------|
| **Festival Automation** | Automatically play Festival stages with OCR verification |
| **Gacha Automation** | Automatically pull gacha with template recognition |
| **Hopping Automation** | Automatically run Pool Hopping with item verification |

---

## Getting Started & Connection

### Step 1: Open Game
1. Launch DOAX Venus Vacation
2. Ensure the game window is fully visible (not minimized)
3. Set game to Windowed or Borderless mode

### Step 2: Run Auto C-Peach
Open **Auto C-Peach.exe** to start the application.

### Step 3: Connect Device
1. Click **"Connect Device"** button at the top right corner
2. Wait for status to change to **"Device Connected"** (green)
3. If connection fails, click **"Refresh"** to retry

### Connection Status
| Status | Color | Meaning |
|--------|-------|---------|
| Not Connected | Red | Device not connected |
| Connecting... | Blue | Connection in progress |
| Device Connected | Green | Successfully connected |
| Connection Failed | Red | Connection failed |


---

## Main Interface

### Layout Overview

```
+-------------------------------------------------------------+
|  Header: Auto C-Peach    [Device Status] [Connect] [Refresh]|
+-------------------------------------------------------------+
|  [Festival] [Gacha] [Hopping] [Settings]  <- Tabs           |
+-----------------------------------+-------------------------+
|                                   |  Progress Panel         |
|  Configuration Panel              |  - Progress bar         |
|  - File Selection                 |  - Statistics           |
|  - Automation Settings            |-------------------------|
|  - Action Buttons                 |  Quick Actions          |
|      [Start] [Pause] [Stop]       |  - Device/OCR test      |
|                                   |-------------------------|
|                                   |  Status                 |
|                                   |  - Current status       |
+-----------------------------------+-------------------------+
|  Activity Logs                    |  Error History          |
+-------------------------------------------------------------+
|  Footer: (c) 2025 Auto C-Peach | Version 1.0      [Status]  |
+-------------------------------------------------------------+
```

### UI Components

#### 1. Pause/Resume Button
- Click **"Pause"** to pause automation
- Click **"Resume"** to continue
- Shortcut: `Ctrl+P`
- Useful when manual intervention is needed

#### 2. Progress Panel
- **Progress Bar**: Progress bar with percentage
- **Statistics**: Total/Success/Failed/Skipped counts
- **Time Info**: Elapsed time, ETA, average time per item
- **Current Item**: Shows item being processed

#### 3. Toast Notifications
Non-blocking notifications at screen corner:
- **Info** (blue): General information
- **Success** (green): Completed successfully
- **Warning** (orange): Warnings
- **Error** (red): Errors

#### 4. Error History Panel
- Displays error history with timestamps
- Categorized by severity (ERROR/WARNING/INFO)
- **Clear** button to clear history

#### 5. Tooltips
Hover over buttons to see guidance.


---

## Festival Automation Tab

### Overview
Festival Automation automates playing Festival stages in the game.

### Operation Flow
1. Touch Event Button
2. Snapshot Before
3. OCR Find & Touch Festival Name (with fallback cache)
4. OCR Find & Touch Rank
5. Snapshot After
6. Pre-Battle Verification
7. Touch Challenge Button
8. Optional: Drag & Drop mini-game
9. Touch OK/Skip buttons
10. Touch Result Button
11. Snapshot Result
12. Post-Battle Verification
13. Close result dialogs

**Note**: Pause/Resume available at any step

### Special Features
- **Resume**: Auto-saves progress, can continue if interrupted
- **Fallback Cache**: Saves successful touch positions for OCR failures
- **Fuzzy Matching**: Handles OCR misreads

### Data File Format (CSV)
```csv
フェス名,フェスランク,推奨ランク,勝利点数,Sランクボーダー,初回クリア報酬,Sランク報酬
Festival Name 1,Rank A,E,1000,5000,Item A,Item B
```

### Steps to Execute
1. **Select Data File**: Browse → Preview
2. **Select Starting Stage**: Dropdown
3. **Resume Options**: Check "Force New Session" to start fresh
4. **Start**: Click **"Start Festival"**
5. **Pause**: Click **"Pause"** or `Ctrl+P`
6. **Stop**: Click **"Stop"** or `Ctrl+Q`/`ESC`/`F9`

---

## Gacha Automation Tab

### Overview
Gacha Automation automates gacha pulls with result verification.

### Operation Flow
1. Find Banner (template matching + auto-scroll)
2. Touch Banner to open
3. Select Pull Type (single/multi)
4. Snapshot Before
5. Confirm Pull
6. Wait for animation
7. Skip Animation (optional)
8. Snapshot After
9. Result Verification (Rarity + Swimsuit)
10. Repeat for remaining pulls

**Note**: Pause/Resume available between pulls

### Templates Structure
```
templates/
└── jp/
    ├── tpl_event.png, tpl_ok.png, tpl_skip.png
    ├── tpl_ssr.png, tpl_sr.png
    └── banners/
        └── banner_name/
            ├── banner.png      # Main banner image
            └── swimsuit.png    # Swimsuit for verification
```

### Steps to Execute
1. **Select Templates Folder**: Browse → Refresh
2. **Configure Pull**: Rarity, Pulls, Type
3. **Add Banners to Queue**
4. **Start**: Start Gacha
5. **Control**: Pause/Resume/Stop


---

## Hopping Automation Tab

### Overview
Hopping Automation automates Pool Hopping with item verification.

### Operation Flow
1. Snapshot Before
2. Touch Use Button
3. Touch OK Button
4. Snapshot Item
5. Verification (OCR scan)
6. Compare with CSV data
7. Record OK/NG/Draw Unchecked
8. Repeat for remaining courses

**Note**: Pause/Resume available at any step

### Verification Results
| Result | Meaning |
|--------|---------|
| **OK** | Correct result, verified |
| **NG** | Incorrect result, verified |
| **Draw Unchecked** | Could not verify (OCR failed) |

### Steps to Execute
1. **Select Data File**: Browse → Preview
2. **Select Starting Course**: Dropdown
3. **Start**: Start Hopping
4. **Control**: Pause/Resume/Stop

---

## Settings Tab

### Log Level
| Level | Description | When to Use |
|-------|-------------|-------------|
| DEBUG | Most detailed | Debugging, finding errors |
| INFO | General information | Normal use |
| WARNING | Warnings only | Reduce log output |
| ERROR | Errors only | Only concerned with errors |

### Performance Settings
| Setting | Description | Recommended |
|---------|-------------|-------------|
| Max Log Lines | Maximum log lines | 1000 |
| Log Poll Interval | Update frequency (ms) | 200 |

---

## Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `Ctrl+Q` | Stop automation (safe) |
| `ESC` | Emergency stop |
| `F9` | Stop automation |
| `Ctrl+P` | Pause/Resume |
| `Ctrl+Enter` | Start automation |

---

## Results & Screenshots

### Result Directory Structure
```
result/
├── festival/
│   ├── snapshots/          # Screenshots
│   ├── results/            # Result files (CSV/JSON/HTML)
│   ├── logs/               # Detailed logs
│   └── .festival_resume.json
├── gacha/
│   ├── snapshots/
│   ├── results/
│   └── logs/
└── hopping/
    ├── snapshots/
    ├── results/
    ├── logs/
    └── .hopping_resume.json
```

### Output Formats
- **CSV**: Easy to open with Excel
- **JSON**: Easy to process with code
- **HTML**: Visual viewing in browser

---

## Troubleshooting

### Common Errors
| Error | Cause | Solution |
|-------|-------|----------|
| Device not connected | Game not open | Open game, keep in foreground |
| OCR failed | Text unclear | Check game resolution |
| Template not found | Image doesn't match | Recapture template |
| Connection timeout | Game not responding | Restart game and tool |

### Optimization Tips
1. **Keep game in foreground** - Don't minimize
2. **Check Error History** - Detect errors early
3. **Use Pause** - Pause when manual intervention needed

---

## FAQ

### Q: When should I use Pause?
A: When manual intervention is needed (e.g., unexpected popup in game).

### Q: How to continue an interrupted session?
A: Tool auto-saves progress. When restarting, it will ask if you want to continue.

### Q: Is my data safe?
A: Tool only reads files you provide and saves results locally. No data sent externally.

---
