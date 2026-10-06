# rtl

Verilog for the feature extraction and HDC scoring pipeline, targeting the iCE40 UP5K first and SKY130 after. Work here starts in Week 4.

The golden model is the Python in `host/mhm/features/` and `host/mhm/models/hdc.py`. It is written in plain numpy with explicit windowing and an LFSR item memory so it can be turned into a bit-exact fixed-point reference before any RTL is written.
