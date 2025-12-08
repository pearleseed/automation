# Auto C-Peach User Manual

## Introduction

Auto C-Peach is an automation tool for DOAX Venus Vacation, supporting:
- **Festival Automation**: Automatically play Festival stages
- **Gacha Automation**: Automatically pull gacha
- **Hopping Automation**: Automatically run Pool Hopping

---

## Getting Started

1. Open DOAX Venus Vacation game
2. Run Auto C-Peach
3. Click **"Connect Device"** at the top right corner
4. Wait until status shows **"Device Connected"** (green)

---

## Festival Automation Tab

### How It Works

The tool automatically performs the following process for each stage:
1. Tap the Event button to open Festival menu
2. Capture screenshot before selecting stage
3. Use OCR to find and tap the Festival name in the list
4. Use OCR to find and tap the corresponding Rank
5. Capture screenshot after selection
6. Verify displayed information (points, rank, rewards) against CSV data
7. Tap Challenge button to start the battle
8. Automatically drag & drop objects if present (mini-game)
9. Tap OK, Skip buttons to complete the battle
10. Capture result screenshot and verify received rewards

**Resume Feature**: If interrupted (app closed, network error...), the tool automatically saves progress and allows continuing from the current stage when restarted.

**Fallback Cache**: If OCR fails to recognize text (due to long text, truncation...), the tool uses saved positions from previous successful taps.

### Preparation
Prepare a CSV/JSON file containing Festival stage information.

### Steps

1. **Select Data File**
   - Click **"Browse"** to select data file
   - Click **"Preview"** to view contents

2. **Select Starting Stage**
   - Choose starting stage from "Start Stage" dropdown
   - Display format: `[Rank] Festival Name - Festival Rank`

3. **Configure Output** (optional)
   - Click **"..."** to choose output location
   - Leave empty for auto-generated file with timestamp

4. **Resume Options**
   - By default, interrupted sessions will auto-resume
   - Check **"Force New Session"** to start fresh

5. **Start**
   - Click **"Start"** to begin
   - If previous session exists, a dialog will ask whether to continue
   - Monitor progress via Progress Panel

6. **Stop**
   - Click **"Stop"** or use hotkeys `Ctrl+Q` / `ESC` / `F9`

---

## Gacha Automation Tab

### How It Works

The tool automatically performs the following process for each pull:
1. Find banner on screen using template matching (image comparison)
2. If not found, automatically scroll down and search again (up to 10 times)
3. Tap the banner to open it
4. Select pull type (Single or Multi 10x)
5. Capture screenshot before pulling
6. Confirm pull and wait for animation
7. Automatically skip animation if possible
8. Capture result screenshot
9. Verify results:
   - Check if correct rarity (SSR/SR) using template matching
   - Check for swimsuit character using template matching
   - If both match -> Save special screenshot marked "SPECIAL"

**Template Matching**: The tool compares screen images with sample images in the banner folder for recognition. Accuracy depends on sample image quality.

### Preparation
Organize templates folder with banner folders:
- Each folder represents one banner
- Contains main banner image for recognition
- Contains swimsuit character images for result verification

### Steps

1. **Select Templates Folder**
   - Click **"Browse"** to select templates folder
   - Click **"Refresh"** to reload banner list
   - Each banner shows preview and image count in folder

2. **Configure Pull Settings**
   - **Rarity**: Select SSR or SR for result verification
   - **Pulls**: Enter number of pulls (1-100)
   - **Type**: Select Single (1 pull) or Multi (10 pulls)

3. **Add Banners to Queue**
   - Click **"Add to Queue"** on desired banner
   - If folder has multiple images, select the main banner image
   - View total banners and pulls in Queue

4. **Manage Queue**
   - **Edit**: Modify pull count, type, rarity of banner
   - **Remove**: Remove banner from queue
   - **Clear**: Clear entire queue

5. **Start**
   - Click **"Start Gacha"**
   - Confirm gacha and pull count in dialog

6. **Stop**
   - Click **"Stop"** or use hotkeys

---

## Hopping Automation Tab

### How It Works

The tool automatically performs the following process for each course:
1. Capture screenshot before using item
2. Tap Use button to use item
3. Tap OK to confirm
4. Capture screenshot of received item
5. Use OCR to read item name and quantity
6. Compare with CSV data to verify correct/incorrect
7. Write results to output file

**Resume Feature**: Similar to Festival, the tool saves progress and allows continuing from the current course.

**OCR Verification**: The tool reads text on screen and compares with expected data in CSV. Supports fuzzy matching to handle cases where OCR misreads a few characters.

### Preparation
Prepare a CSV/JSON file containing course information.

### Steps

1. **Select Data File**
   - Click **"Browse"** to select file
   - Click **"Preview"** to view contents

2. **Select Starting Course**
   - Choose course from "Start Course" dropdown

3. **Configure Output** (optional)
   - Click **"..."** to choose output location

4. **Resume Options**
   - Check **"Force New Session"** to start fresh

5. **Start**
   - Click **"Start"**

6. **Stop**
   - Click **"Stop"** or use hotkeys

---

## Settings Tab

- **Log Level**: Adjust log detail level
  - DEBUG: Show all detailed information
  - INFO: General information (recommended)
  - WARNING: Show warnings only
  - ERROR: Show errors only

- **Max Log Lines**: Maximum log lines displayed (reduce for better performance)
- **Log Poll Interval**: Log update frequency (increase to reduce CPU load)

---

## Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `Ctrl+Q` | Stop automation |
| `ESC` | Emergency stop |
| `F9` | Stop automation |

---

## Results & Screenshots

All results are saved in the `result/` folder:
- **Snapshots**: Screenshots captured at important steps
- **Results**: CSV/JSON files containing verification results
- **Logs**: Detailed log files of the running process

Each automation has its own folder (festival/, gacha/, hopping/) for easy management.

---

## Important Notes

- **Do not move or minimize** the game window while running
- **Keep game in foreground** so the tool can capture screen and control
- Use **Preview** to verify data before running
- If stuck, use **ESC** for emergency stop
- Check **Activity Log** at the bottom of the screen to monitor status and errors
- You can drag the divider between tabs and log to adjust sizes
