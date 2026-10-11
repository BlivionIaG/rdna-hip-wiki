# Format silicon notes

Silicon-side notes for EXL3, DSv4 Flash mxfp4, and the W4A16 prefill K_STEP config. Format contracts live in [../../kernels/](../../kernels/README.md).

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [exl3.md](exl3.md) | EXL3/QTIP: Viterbi is quant-time; infer is 3-inst codebook → half → `fdot2`. mul1/K=1..8 lock @ `30b0bd4e` |
| [dsv4-flash.md](dsv4-flash.md) | DSv4 Flash: mxfp4 expert tile is the decode kernel; leftover is MXFP8 + e8m0/128², not W8A16-FP8 |
| [w4a16-prefill-config.md](w4a16-prefill-config.md) | ConfigA K_STEP=32 dest; ConfigH Leave (`7ac98a26`); K_STEP split repair @ `3a0786ea` (8 repairs / 13 legacy keep / 0 deviation) |
