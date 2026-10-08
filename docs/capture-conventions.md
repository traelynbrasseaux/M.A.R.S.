# Capture conventions

File naming and the `captures.csv` columns are defined in [interface.md](interface.md#capture-files). This page fixes the vocabulary and the recording routine so captures from different days can be compared.

## Vocabulary

Tokens are lowercase `a-z`, `0-9`, and `-`. Add new values here before using them.

**machine:** the physical machine plus an index, e.g. `fan1`, `pump1`. Use `sim` for `mhm simulate` output.

**condition:**

| Token        | Meaning |
|--------------|---------|
| `normal`     | healthy machine, sensor properly mounted |
| `unbalance`  | added mass on the rotor; put the mass and radius in `--notes` |
| `loose`      | sensor or machine mount deliberately loosened; say which in `--notes` |
| `clog`       | pump intake partially blocked; estimate the blocked fraction in `--notes` |
| `off`        | machine powered off, for the ambient noise floor |

**speed:** nominal shaft speed, e.g. `1500rpm`, or `var` when it is swept during the run. Use `0rpm` with `off`.

## Recording routine

1. Mount the sensor plate at the marked spot and photograph it once per session. Only the rigid ADXL345 plate goes on the machine; the breadboard and ESP32 sit off it, on wires under about 20 cm.
2. Let the machine run for 2 minutes before recording so temperature and speed settle.
3. Record 120 s per run (`--duration 120`).
4. Record at least 3 `normal` runs per speed, and remount the node between at least two of them so the model learns mounting variation rather than one exact placement.
5. Record an `off` run at the start of each session.
6. Check the live summary while recording. Any CRC errors or dropped frames go in `--notes`.
