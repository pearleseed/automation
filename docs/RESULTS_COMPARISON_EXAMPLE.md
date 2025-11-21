# Results Comparison: Before vs After

## Before Implementation

### CSV Output (Old Format)
```csv
test_case_id,フェス名,推奨ランク,勝利点数,result,timestamp,error_message
1,春祭り,B,1000,NG,2024-11-21 10:30:00,Verification failed
2,夏祭り,A,2000,OK,2024-11-21 10:32:00,
3,秋祭り,C,500,NG,2024-11-21 10:34:00,Verification failed
```

### Problems
- ❌ Can't see which specific field failed
- ❌ No visibility into extracted vs expected values
- ❌ Difficult to identify patterns in failures
- ❌ Can't distinguish between pre-battle and post-battle failures

---

## After Implementation

### CSV Output (New Format)
```csv
test_case_id,フェス名,推奨ランク,result,pre_勝利点数_status,pre_勝利点数_expected,pre_勝利点数_extracted,pre_推奨ランク_status,pre_推奨ランク_expected,pre_推奨ランク_extracted,post_獲得ザックマネー_status,post_獲得ザックマネー_expected,post_獲得ザックマネー_extracted,timestamp
1,春祭り,B,NG,OK,1000,1000,NG,B,C,OK,500,500,2024-11-21 10:30:00
2,夏祭り,A,OK,OK,2000,2000,OK,A,A,OK,1000,1000,2024-11-21 10:32:00
3,秋祭り,C,NG,OK,500,500,OK,C,C,NG,200,150,2024-11-21 10:34:00
```

### Benefits
- ✅ See exactly which field failed (推奨ランク in stage 1, 獲得ザックマネー in stage 3)
- ✅ Compare expected vs extracted values (B vs C, 200 vs 150)
- ✅ Identify patterns (e.g., rank detection issues)
- ✅ Distinguish pre-battle vs post-battle failures
- ✅ Better data for analysis and debugging

---

## Log Output Comparison

### Before
```
[INFO] Stage 1: 春祭り
[INFO] Pre-Battle Verification: ✗ 4/5 matched (推奨ランク:C≠B)
[ERROR] Step 6 failed after 5 retries
[ERROR] Stage 1 failed
```

### After
```
[INFO] Stage 1: 春祭り
[INFO] PRE-BATTLE VERIFICATION
[INFO]   ✓ 勝利点数: MATCH (expected: 1000, extracted: 1000)
[INFO]   ✗ 推奨ランク: MISMATCH (expected: B, extracted: C)
[INFO]   ✓ Sランクボーダー: MATCH (expected: 5000, extracted: 5000)
[INFO]   ✓ 初回クリア報酬: MATCH (expected: ポーション, extracted: ポーション)
[INFO]   ✓ Sランク報酬: MATCH (expected: エリクサー, extracted: エリクサー)
[INFO] Verification: ✗ 4/5 matched (推奨ランク:C≠B)
[INFO] BATTLE EXECUTION
[INFO] POST-BATTLE VERIFICATION
[INFO]   ✓ 獲得ザックマネー: MATCH (expected: 500, extracted: 500)
[INFO]   ✓ 獲得アイテム: MATCH (expected: ポーション, extracted: ポーション)
[INFO]   ✓ 獲得EXP-Ace: MATCH (expected: 100, extracted: 100)
[INFO] Stage 1 completed (NG due to pre-battle mismatch)
```

---

## Analysis Examples

### Example 1: Find all stages where rank detection failed
```python
import pandas as pd

df = pd.read_csv("results.csv")
rank_failures = df[df["pre_推奨ランク_status"] == "NG"]
print(f"Rank detection failed in {len(rank_failures)} stages")
print(rank_failures[["test_case_id", "フェス名", "pre_推奨ランク_expected", "pre_推奨ランク_extracted"]])
```

### Example 2: Calculate accuracy per field
```python
fields = ["勝利点数", "推奨ランク", "Sランクボーダー", "獲得ザックマネー"]
for field in fields:
    col = f"pre_{field}_status" if field in ["勝利点数", "推奨ランク", "Sランクボーダー"] else f"post_{field}_status"
    if col in df.columns:
        accuracy = (df[col] == "OK").sum() / len(df) * 100
        print(f"{field}: {accuracy:.1f}% accuracy")
```

### Example 3: Identify common OCR mistakes
```python
# Find cases where extracted value differs from expected
mismatches = df[df["pre_推奨ランク_status"] == "NG"]
for _, row in mismatches.iterrows():
    print(f"Expected: {row['pre_推奨ランク_expected']} → Got: {row['pre_推奨ランク_extracted']}")
```

---

## Summary

The new detailed results format provides:
1. **Granular visibility** into each verification field
2. **Continuous execution** without stopping on failures
3. **Rich data** for analysis and debugging
4. **Better insights** into OCR/detection accuracy
5. **Easier troubleshooting** of specific issues
