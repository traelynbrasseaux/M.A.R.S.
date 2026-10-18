# Evaluation results

Newest entry first. Each entry records the code it was run on and what hardware it does or does not cover.

## 2026-10-18: pre-hardware baseline

**Status: pre-hardware.** Code at commit `708a4e8`. Martin has the ESP32 board wired to both sensors and streaming, but no recordings from the real rig are in these results, and the on-device ML firmware has not yet run on the board. Every number below comes from public datasets or the simulator, computed on a PC, both in Python (`host/mhm`) and with the node's C++ (`firmware/node/lib/ml`) built for the PC.

### What was evaluated

| Model | How it scores a window | Why it is here |
|-------|------------------------|----------------|
| Mahalanobis | Distance from the average normal feature vector, in units of normal variation, including how features vary together | Classical baseline that HDC has to match |
| HDC binary | Each feature rounded to one of 16 levels, encoded as 4,096-bit patterns, merged by majority vote; Hamming distance to one normal prototype | Hardware target: only XOR, counting and comparing |
| HDC bipolar | Same encoding, but keeps vote totals instead of rounding them; cosine distance to the prototype | Shows what binary's rounding gives up |

- **Features:** 1,024-sample windows every 512 samples. Vibration gives 33 numbers per window: per axis, 8 log band energies from 5 to 1,600 Hz, RMS, kurtosis and crest factor. Audio gives 45: 32 log mel band energies and 13 MFCCs.
- **Protocol:** models learn from normal data only. Normal recordings are dealt out whole (a clip, a record, or a capture run) into training, calibration and test sets, so no recording is split across them. A recording's score is the mean of its window scores. The alarm threshold is the 95th percentile of the calibration recordings' scores. The test set is the held-out normals plus all faulty recordings. Split seed 0.
- **Backends:** `python` runs `host/mhm`. `device` runs the node's C++, which learns as the node does: feature ranges from the first half of the training windows and the HDC prototype from the second half. Mahalanobis uses all of them.

### Metrics

- **AUC:** the chance that a random faulty recording scores higher than a random normal one. 1.0 separates perfectly, 0.5 is a coin flip. It does not depend on the threshold.
- **pAUC:** AUC over false positive rates of 10% or less, scaled to 0 to 1 (DCASE 2020 definition, no McClish correction).
- **Recall** (detection rate) and **FPR** (false alarm rate) at the calibration threshold.
- **Precision, F1, accuracy** at the same threshold. These depend on the test mix, which here is 50% to 57% faulty, far more than a real machine sees. The TP / FN / FP / TN counts let you recompute any of them.

### Datasets and how they apply

| Dataset | What it is | Subset used | Tests our target fault | Caveats |
|---------|------------|-------------|------------------------|---------|
| MIMII (Hitachi, 2019) | Factory recordings of pumps, fans, valves and slide rails, normal and faulty, with background noise mixed in at 6, 0 and -6 dB | Pump `id_00`, 6 dB, microphone channel 0 at 16 kHz: 150 normal and 100 abnormal clips of 10 s | Pump intake clog (MIMII pump faults include clogging, leakage and contamination), audio stream | Lab microphone array, not a mic on the machine; the cleanest noise level; the first 150 normal files, likely one session; fault types not labelled separately |
| MAFAULDA (UFRJ) | Bench rig with a motor-driven rotor on two bearings, industrial accelerometers at 50 kHz, faults set up by hand | Underhang bearing, 3 axes, downsampled to 3,125 Hz to match the ADXL345: all 49 normal records plus 6 imbalance records each at 6, 20 and 35 g, 5 s each | Rotor unbalance, vibration stream; the rig is close to Martin's motor rig | Lab accelerometers far quieter than our MEMS part; shaft speed varies from about 12 to 61 Hz across records; small test set |
| Simulator (`mhm simulate`) | Our frame format with a 25 Hz shaft, vibration at 1x, 2x and 3x the shaft speed, blade-pass tones and noise; unbalance makes 1x 4 times stronger | 5 normal and 2 unbalance runs of 60 s, cut into 5 s segments | Pipeline checks only | Faults are obvious by construction |

No dataset covers loose mounting; that needs our own captures.

### Results

**MIMII pump, 6 dB, `id_00`** (train 60, calibration 15, test 75 normal + 100 abnormal)

| Backend | Model | AUC | pAUC | Recall | FPR | Precision | F1 | Accuracy | TP / FN / FP / TN |
|---------|-------|-----|------|--------|-----|-----------|----|----------|-------------------|
| python | Mahalanobis | 0.992 | 0.972 | 0.98 | 0.15 | 0.90 | 0.94 | 0.93 | 98 / 2 / 11 / 64 |
| python | HDC binary | 0.999 | 0.995 | 1.00 | 0.13 | 0.91 | 0.95 | 0.94 | 100 / 0 / 10 / 65 |
| python | HDC bipolar | 1.000 | 0.996 | 1.00 | 0.09 | 0.93 | 0.97 | 0.96 | 100 / 0 / 7 / 68 |
| device | Mahalanobis | 0.992 | 0.972 | 0.98 | 0.15 | 0.90 | 0.94 | 0.93 | 98 / 2 / 11 / 64 |
| device | HDC binary | 0.999 | 0.995 | 1.00 | 0.09 | 0.93 | 0.97 | 0.96 | 100 / 0 / 7 / 68 |
| device | HDC bipolar | 0.999 | 0.995 | 1.00 | 0.11 | 0.93 | 0.96 | 0.95 | 100 / 0 / 8 / 67 |

**MAFAULDA imbalance** (train 25, calibration 6, test 18 normal + 18 abnormal)

| Backend | Model | AUC | pAUC | Recall | FPR | Precision | F1 | Accuracy | TP / FN / FP / TN |
|---------|-------|-----|------|--------|-----|-----------|----|----------|-------------------|
| python | Mahalanobis | 0.963 | 0.944 | 0.83 | 0.00 | 1.00 | 0.91 | 0.92 | 15 / 3 / 0 / 18 |
| python | HDC binary | 0.920 | 0.833 | 0.56 | 0.00 | 1.00 | 0.71 | 0.78 | 10 / 8 / 0 / 18 |
| python | HDC bipolar | 0.914 | 0.833 | 0.50 | 0.00 | 1.00 | 0.67 | 0.75 | 9 / 9 / 0 / 18 |
| device | Mahalanobis | 0.963 | 0.944 | 0.83 | 0.00 | 1.00 | 0.91 | 0.92 | 15 / 3 / 0 / 18 |
| device | HDC binary | 0.914 | 0.833 | 0.61 | 0.00 | 1.00 | 0.76 | 0.81 | 11 / 7 / 0 / 18 |
| device | HDC bipolar | 0.920 | 0.802 | 0.44 | 0.00 | 1.00 | 0.62 | 0.72 | 8 / 10 / 0 / 18 |

**Simulated captures** (train 24, calibration 12, test 24 normal + 24 unbalance segments of 5 s)

| Stream | Backend | Model | AUC | pAUC | Recall | FPR | Precision | F1 | Accuracy | TP / FN / FP / TN |
|--------|---------|-------|-----|------|--------|-----|-----------|----|----------|-------------------|
| accel | python | Mahalanobis | 1.000 | 1.000 | 1.00 | 0.12 | 0.89 | 0.94 | 0.94 | 24 / 0 / 3 / 21 |
| accel | python | HDC binary | 1.000 | 1.000 | 1.00 | 0.33 | 0.75 | 0.86 | 0.83 | 24 / 0 / 8 / 16 |
| accel | python | HDC bipolar | 1.000 | 1.000 | 1.00 | 0.25 | 0.80 | 0.89 | 0.88 | 24 / 0 / 6 / 18 |
| accel | device | Mahalanobis | 1.000 | 1.000 | 1.00 | 0.12 | 0.89 | 0.94 | 0.94 | 24 / 0 / 3 / 21 |
| accel | device | HDC binary | 1.000 | 1.000 | 1.00 | 0.29 | 0.77 | 0.87 | 0.85 | 24 / 0 / 7 / 17 |
| accel | device | HDC bipolar | 1.000 | 1.000 | 1.00 | 0.25 | 0.80 | 0.89 | 0.88 | 24 / 0 / 6 / 18 |
| audio | python | Mahalanobis | 1.000 | 1.000 | 1.00 | 0.17 | 0.86 | 0.92 | 0.92 | 24 / 0 / 4 / 20 |
| audio | python | HDC binary | 1.000 | 1.000 | 1.00 | 0.08 | 0.92 | 0.96 | 0.96 | 24 / 0 / 2 / 22 |
| audio | python | HDC bipolar | 1.000 | 1.000 | 1.00 | 0.04 | 0.96 | 0.98 | 0.98 | 24 / 0 / 1 / 23 |
| audio | device | Mahalanobis | 1.000 | 1.000 | 1.00 | 0.17 | 0.86 | 0.92 | 0.92 | 24 / 0 / 4 / 20 |
| audio | device | HDC binary | 1.000 | 1.000 | 1.00 | 0.08 | 0.92 | 0.96 | 0.96 | 24 / 0 / 2 / 22 |
| audio | device | HDC bipolar | 1.000 | 1.000 | 1.00 | 0.08 | 0.92 | 0.96 | 0.96 | 24 / 0 / 2 / 22 |

### C++ port checks

These run in `uv run pytest`, which builds `lib/ml` for the PC first.

| Check | Measured | Test tolerance |
|-------|----------|----------------|
| Vibration features vs Python | max abs difference 1.8e-6 | 1e-4 |
| Audio features vs Python | max abs difference 6.0e-7 | 1e-4 |
| HDC binary scores | identical, bit for bit | exact |
| HDC bipolar scores | max abs difference 3.7e-9 | 1e-6 |
| Mahalanobis scores | max relative difference 4.9e-5 | 1e-3 |

The node's full detector (learning, calibration and 1 s detection blocks) was replayed on the PC with `learn 60 30` and the default margin of 1.2, over 130 s of normal simulated data followed by 30 s of unbalance:

- **No false alarms:** 0 alarms in 42 normal accel blocks and 40 normal audio blocks, for all three models.
- **Full detection:** every unbalance block was flagged, 30 of 30 accel and 29 of 29 audio, for all three models.
- **Reaction time:** `mhm replay` on a capture whose unbalance starts at 100 s, run with `--learn 40 --calibrate 20`, raised the first alert between 100.5 and 101.5 s, with none before.

### Findings

1. **The C++ matches the Python.** Mahalanobis gives identical results on both datasets. HDC is within 0.006 AUC, a difference that comes from the node's two-phase learning.
2. **All three models separate the MIMII pump faults** (AUC 0.99 or better), with HDC slightly ahead.
3. **On MAFAULDA, Mahalanobis leads** (0.963 against 0.91 to 0.92). A likely cause, not yet tested: the records span shaft speeds of about 12 to 61 Hz, and one averaged HDC prototype represents a multi-speed "normal" less well than a covariance model does. Prototypes per speed range would test that.
4. **Thresholds, more than separation, limit recall and false alarms.** MAFAULDA recall was 0.44 to 0.83 at zero false alarms, because the threshold came from only 6 calibration records. On the simulator, false alarms ran from 4% to 33% in this table, and from 0% to 33% across split seeds, depending on which run calibrated.
5. **False alarms per hour is the number that matters on the real machine.** A false alarm rate near 10% on 10 s clips would mean about 30 false alarms an hour.

### Not yet measured (needs the board)

- **Memory headroom:** the ML needs about 200 KB of the ESP32's roughly 300 KB heap, estimated from object sizes. The node's 10 s health log reports free and largest-block heap.
- **ML speed:** about 20% of one core, estimated. The health log reports the worst processing time per frame and any dropped ML blocks.
- **Real-rig performance:** false alarms per hour after remounting, and detection of unbalance, loose mounting and pump clogs.
- **Threshold margin:** what margin works on real captures, tried with `mhm replay` on recorded runs.

### Reproducing

```bash
cd host
uv run python scripts/download_mimii.py
uv run python scripts/download_mafaulda.py
uv run python scripts/evaluate.py mimii                      # add --backend device for the C++
uv run python scripts/evaluate.py mafaulda
uv run python scripts/evaluate.py captures --stream accel    # after the simulator loop in the README
uv run python scripts/evaluate.py captures --stream audio
```

`evaluate.py` prints AUC, pAUC, recall and false alarm rate, and saves ROC and score plots to `host/results/`. Precision, F1 and accuracy above were computed from the TP / FN / FP / TN counts at the same thresholds.
