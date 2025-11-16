# Fuzzy Matching Guide

## Overview

This document explains the fuzzy matching system implemented in the automation framework. Fuzzy matching allows the system to find and interact with text even when there are OCR errors, typos, or slight variations.

**Date**: November 16, 2025

---

## 🎯 What is Fuzzy Matching?

Fuzzy matching is an approximate string matching technique that can find text even when:
- **OCR makes errors**: "イベント" might be recognized as "イヘント"
- **Text has variations**: "Festival Event" vs "Festival-Event"  
- **Partial matches occur**: Searching for "Event" in "Special Event 2024"
- **There are typos**: Small character differences don't prevent matching

---

## 🔧 How It Works

### Core Components

1. **TextProcessor** (`core/detector.py`)
   - `normalize_text()`: Normalizes text for comparison (removes spaces, lowercase, etc.)
   - `calculate_similarity()`: Calculates similarity score between two strings (0.0-1.0)
   - `fuzzy_match()`: Determines if two texts match above a threshold

2. **find_text()** (`core/base.py`)
   - Updated to use fuzzy matching by default
   - Finds best match above threshold from multiple OCR results
   - Returns match with similarity score

3. **find_and_touch_in_roi()** (`core/base.py`)
   - Uses fuzzy matching to find text in ROI
   - Touches the found text location
   - Configurable threshold and fuzzy mode

---

## 📝 Usage

### Basic Usage (Default - Fuzzy Enabled)

```python
# Fuzzy matching enabled by default with threshold 0.7
automation.find_and_touch_in_roi('フェス名', 'イベント')
```

### Custom Threshold

```python
# Stricter matching (higher threshold)
automation.find_and_touch_in_roi('フェス名', 'イベント', threshold=0.9)

# More lenient matching (lower threshold)
automation.find_and_touch_in_roi('フェス名', 'イベント', threshold=0.6)
```

### Exact Matching (Disable Fuzzy)

```python
# Use exact substring matching instead
automation.find_and_touch_in_roi('フェス名', 'イベント', use_fuzzy=False)
```

---

## ⚙️ Configuration

### Global Configuration

Edit `core/config.py` to set default fuzzy matching behavior:

```python
FESTIVAL_CONFIG: Dict[str, Any] = {
    # ...
    
    # Fuzzy matching config
    'fuzzy_matching': {
        'enabled': True,      # Enable/disable fuzzy matching
        'threshold': 0.7,     # Default similarity threshold
    },
    
    # ...
}
```

### Threshold Guidelines

| Threshold | Strictness | Use Case |
|-----------|------------|----------|
| 0.9 - 1.0 | Very Strict | Perfect match required, minimal OCR errors |
| 0.8 - 0.9 | Strict | Good OCR quality, exact words expected |
| 0.7 - 0.8 | **Balanced** | **Recommended default**, handles common OCR errors |
| 0.6 - 0.7 | Lenient | Poor OCR quality, significant variations |
| 0.5 - 0.6 | Very Lenient | Extreme OCR errors, use with caution |

---

## 🔍 How Matching Works

### Similarity Calculation

The system uses the Gestalt Pattern Matching algorithm (SequenceMatcher) to calculate similarity:

```python
similarity = SequenceMatcher(None, text1, text2).ratio()
# Returns: 0.0 (completely different) to 1.0 (identical)
```

### Matching Process

1. **Normalize both texts**: Remove spaces, punctuation, convert to lowercase
2. **Check exact match**: If normalized texts are identical → similarity = 1.0
3. **Check substring match**: If one contains the other → similarity = 0.9
4. **Calculate similarity**: Use SequenceMatcher for partial similarity
5. **Compare to threshold**: Match if similarity ≥ threshold

### Example

```
Search: "イベント"
OCR Result: "イヘント"  (OCR error: ベ → ヘ)

Normalized:
  - Search: "いべんと"
  - OCR: "いへんと"

Similarity calculation:
  - Character-by-character comparison
  - 3 out of 4 characters match
  - Similarity ≈ 0.75

Result: MATCH (0.75 ≥ 0.7 threshold)
```

---

## 📊 Logging and Debugging

### Log Output

When fuzzy matching finds text, it logs:

```
INFO | Find & touch 'イベント' in ROI 'フェス名' (fuzzy matching)
DEBUG | OCR found 3 text(s) in ROI 'フェス名': ['イヘント', 'ランクE', '開始']
DEBUG | Fuzzy match: 'イヘント' ~ 'イベント' (similarity: 0.75)
INFO | ✓ Found 'イヘント' ~ 'イベント' in ROI 'フェス名' (similarity: 0.75) at (100, 200)
```

### Debug Mode

To see detailed matching information, set log level to DEBUG:

```python
import logging
logging.getLogger('core.base').setLevel(logging.DEBUG)
```

---

## 🎨 Advanced Usage

### Per-Call Configuration

You can override global settings for specific calls:

```python
# High-confidence match required for critical actions
automation.find_and_touch_in_roi(
    'フェス名', 
    'イベント',
    threshold=0.95,  # Very strict
    use_fuzzy=True
)

# Lenient match for known OCR issues
automation.find_and_touch_in_roi(
    'フェスランク',
    'E',
    threshold=0.6,   # Lenient
    use_fuzzy=True
)

# Exact match for precise text
automation.find_and_touch_in_roi(
    'ボタン',
    'OK',
    use_fuzzy=False  # Exact only
)
```

### Custom Matching Logic

For advanced scenarios, use TextProcessor directly:

```python
from core.detector import TextProcessor

# Check if two texts match
is_match = TextProcessor.fuzzy_match('イベント', 'イヘント', threshold=0.7)

# Calculate exact similarity
similarity = TextProcessor.calculate_similarity('イベント', 'イヘント')

# Normalize text
normalized = TextProcessor.normalize_text('  Special Event  ')
# Returns: "specialevent"
```

---

## 💡 Best Practices

### 1. Start with Default Settings
```python
# Use defaults first (threshold=0.7, fuzzy=True)
find_and_touch_in_roi('フェス名', 'イベント')
```

### 2. Adjust Threshold Based on Results

If getting **false positives** (wrong text matched):
```python
# Increase threshold
find_and_touch_in_roi('フェス名', 'イベント', threshold=0.8)
```

If getting **false negatives** (correct text not found):
```python
# Decrease threshold
find_and_touch_in_roi('フェス名', 'イベント', threshold=0.6)
```

### 3. Use Exact Matching for Short Text
```python
# For single characters or very short text, exact matching may be better
find_and_touch_in_roi('ランク', 'E', use_fuzzy=False)
```

### 4. Test with Real OCR Data

Always test with actual screenshots to tune thresholds:

```python
# Capture OCR results
ocr_results = automation.ocr_roi_with_lines('フェス名')

# Test different thresholds
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    match = automation.find_text(ocr_results, 'イベント', threshold=threshold)
    if match:
        print(f"Threshold {threshold}: Found '{match['text']}' (sim: {match['similarity']:.2f})")
```

---

## 🚨 Troubleshooting

### Problem: Text Not Found

**Symptoms**: "Text 'X' not found in ROI 'Y'"

**Solutions**:
1. **Lower threshold**: Try 0.6 instead of 0.7
2. **Check OCR results**: Enable DEBUG logging to see what OCR detected
3. **Verify ROI**: Ensure ROI contains the text
4. **Try exact matching**: Set `use_fuzzy=False` to test if text exists

### Problem: Wrong Text Matched

**Symptoms**: Touches wrong location, matches similar but incorrect text

**Solutions**:
1. **Raise threshold**: Try 0.8 or 0.9
2. **Use exact matching**: Set `use_fuzzy=False`
3. **Refine search text**: Use more specific text
4. **Check OCR quality**: Improve screenshot quality if possible

### Problem: Inconsistent Results

**Symptoms**: Sometimes works, sometimes doesn't

**Solutions**:
1. **Increase retry attempts**: Retry OCR multiple times
2. **Wait longer**: Add delay before OCR to let screen stabilize
3. **Check screen resolution**: Ensure consistent device resolution
4. **Use detector mode**: Try YOLO detector instead of OCR-only

---

## 📈 Performance Considerations

### Caching

TextProcessor uses `@lru_cache` for performance:
- `normalize_text()`: Caches 1024 entries
- `calculate_similarity()`: Caches 512 entries

This means repeated comparisons are extremely fast.

### Optimization Tips

1. **Limit OCR results**: Only OCR relevant ROIs
2. **Use pre-filtering**: Filter by text length before fuzzy matching
3. **Batch operations**: Process multiple ROIs in one screenshot
4. **Configure max cache size**: Adjust LRU cache size if needed

---

## 🔄 Migration from Old Code

### Before (Exact Matching Only)

```python
# Old code - substring match only
def find_text(self, ocr_results, search_text):
    search_lower = search_text.lower().strip()
    for result in ocr_results:
        if search_lower in result['text'].lower():
            return result
    return None
```

### After (Fuzzy Matching)

```python
# New code - fuzzy matching with fallback
def find_text(self, ocr_results, search_text, threshold=0.7, use_fuzzy=True):
    if use_fuzzy:
        # Find best match above threshold
        best_match = None
        best_similarity = 0.0
        
        for result in ocr_results:
            similarity = calculate_similarity(normalize(result['text']), normalize(search_text))
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = result
        
        return best_match
    else:
        # Fallback to exact matching
        # ... (same as old code)
```

### Backward Compatibility

All existing code continues to work:
- Default parameters enable fuzzy matching
- Old exact behavior available with `use_fuzzy=False`
- No breaking changes

---

## 📚 Related Documentation

- **REFACTORING_SUMMARY.md**: Overview of all refactoring changes
- **detector.py**: TextProcessor implementation details
- **base.py**: find_text() and find_and_touch_in_roi() implementations

---

## 🧪 Testing Examples

### Test 1: Basic Fuzzy Matching

```python
from core.detector import TextProcessor

# Test various OCR errors
test_cases = [
    ("イベント", "イベント", 1.0),      # Perfect match
    ("イベント", "イヘント", 0.75),     # OCR error
    ("イベント", "イペント", 0.75),     # Similar error
    ("イベント", "イ", 0.25),           # Partial match
    ("Event", "Event2024", 0.7),       # Substring
]

for search, ocr, expected_sim in test_cases:
    actual = TextProcessor.calculate_similarity(
        TextProcessor.normalize_text(search),
        TextProcessor.normalize_text(ocr)
    )
    print(f"'{search}' vs '{ocr}': {actual:.2f} (expected ~{expected_sim})")
```

### Test 2: Threshold Tuning

```python
# Test different thresholds
search_text = "イベント"
ocr_texts = ["イヘント", "イペント", "イベン", "Event"]

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"\nThreshold: {threshold}")
    for ocr_text in ocr_texts:
        match = TextProcessor.fuzzy_match(search_text, ocr_text, threshold)
        sim = TextProcessor.calculate_similarity(
            TextProcessor.normalize_text(search_text),
            TextProcessor.normalize_text(ocr_text)
        )
        status = "✓ MATCH" if match else "✗ NO MATCH"
        print(f"  {status} '{ocr_text}' (similarity: {sim:.2f})")
```

---

## ✅ Summary

**Fuzzy matching has been implemented to:**

1. ✅ Handle OCR errors gracefully
2. ✅ Find approximate text matches
3. ✅ Improve automation reliability
4. ✅ Provide configurable thresholds
5. ✅ Maintain backward compatibility
6. ✅ Log detailed matching information
7. ✅ Optimize with caching

**Key takeaways:**

- **Default threshold: 0.7** (balanced)
- **Fuzzy enabled by default** (can disable per call)
- **Configurable globally** (in config.py)
- **Detailed logging** (enable DEBUG for details)
- **Backward compatible** (old code still works)

---

**Last Updated**: November 16, 2025

