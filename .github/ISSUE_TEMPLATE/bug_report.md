---
name: Bug report
about: Report a defect in MedAugment
title: "[bug] "
labels: bug
---

## Description
A clear description of what is going wrong.

## Reproduction
Minimal Python snippet, including a stand-in `MedVolume` (shape, dtype,
spacing) that triggers the issue.

```python
import numpy as np
from medaugment import MedVolume, Compose
from medaugment.transforms import RandomAffine

vol = MedVolume(image=np.zeros((32, 64, 64), dtype=np.float32), spacing=(1, 1, 1))
out = Compose([RandomAffine(rotation=15)], seed=0)(vol)
```

## Expected vs actual
- Expected: …
- Actual: …

## Environment
- MedAugment version: `pip show medaugment`
- Python version:
- OS:
- Modality / vendor (if relevant):

## Traceback
```
<paste full traceback here>
```
