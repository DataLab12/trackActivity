# Clicks
A utility for quickly and easily selecting and saving lists of pixel coordinates.

## Requirements:
```bash
pip install pyperclip
```

## Usage:
```bash
python clicks.py (optional)[image file path]
```
 If a valid image file path is not provided, a file selection dialog will appear.

## User inputs:
 - Double-click: Draws a cross at the point selected and displays coordinates 
 - 's'-key: Save last selected point to output array
 - backspace: Remove last recorded point from output array
 - return: Exit [output array will be saved to clipboard and printed to console]
