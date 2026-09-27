| set | cache | n | median APE | p90 APE | threshold (median / p90) | verdict |
|---|---|---|---|---|---|---|
| V1 | cold | 594 | 9.0% | 22.5% | 15% / 30% | PASS |
| V1 | warm | 594 | 2.3% | 13.3% | 15% / 30% | PASS |
| V5 | cold | 882 | 12.3% | 32.1% | 22% / 45% | PASS |
| V6 | cold | 1971 | 11.1% | 20.2% | 22% / 45% | PASS |

| set | cache | groups | model median / max regret | heuristic median / max regret | verdict (≤ 5 % / 15 %) |
|---|---|---|---|---|---|
| V1 | cold | 66 | 0.0% / 24.3% | 0.8% / 10.0% | FAIL |
| V1 | warm | 66 | 0.0% / 41.7% | 1.0% / 41.7% | FAIL |
| V5 | cold | 98 | 0.0% / 66.3% | 1.2% / 15.1% | FAIL |
| V6 | cold | 219 | 0.0% / 24.5% | 64.2% / 1152.7% | FAIL |

Worst cells, V1 cold:

- `decode_B2_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal` fd_s2: measured 15.9 µs, predicted 26.1 µs
- `decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal` fd_s4: measured 16.4 µs, predicted 26.1 µs
- `decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal` fd_s4_paged: measured 16.8 µs, predicted 26.5 µs
- `decode_B2_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal` fd_s2_paged: measured 16.8 µs, predicted 26.5 µs
- `decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal` fd_s2_paged: measured 22.3 µs, predicted 35.1 µs

Worst cells, V1 warm:

- `decode_B32_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal` fa2: measured 57.9 µs, predicted 32.5 µs
- `decode_B32_Lq1_Lkv512_Hq32_Hkv8_d128_float16_causal` flashdecoding: measured 57.9 µs, predicted 32.5 µs
- `decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal` fd_s128: measured 28.1 µs, predicted 17.5 µs
- `decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal` fd_s128_paged: measured 28.2 µs, predicted 17.7 µs
- `decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal` fd_s128: measured 34.0 µs, predicted 22.5 µs

Worst cells, V5 cold:

- `decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal` fd_s4: measured 11.2 µs, predicted 22.0 µs
- `decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal` flashdecoding: measured 11.2 µs, predicted 22.0 µs
- `decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal` fd_s4_paged: measured 11.3 µs, predicted 22.1 µs
- `decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal` flashdecoding_paged: measured 11.4 µs, predicted 22.1 µs
- `decode_B1_Lq1_Lkv512_Hq28_Hkv4_d128_float16_causal` fd_s8: measured 11.6 µs, predicted 22.1 µs

Worst cells, V6 cold:

- `decode_B32_Lq1_Lkv16384+512x31_Hq32_Hkv8_d128_float16_causal` flashdecoding: measured 1869.9 µs, predicted 906.6 µs
- `decode_B32_Lq1_Lkv16384x4+512x28_Hq32_Hkv8_d128_float16_causal` fa2: measured 1841.1 µs, predicted 906.6 µs
- `decode_B32_Lq1_Lkv32768x4+512x28_Hq32_Hkv8_d128_float16_causal` fa2: measured 3547.8 µs, predicted 1754.9 µs
- `decode_B32_Lq1_Lkv16384+512x31_Hq32_Hkv8_d128_float16_causal` fa2: measured 1832.4 µs, predicted 906.6 µs
- `decode_B26_Lq1_Lkv16384x4+1024x22_Hq32_Hkv8_d128_float16_causal` fa2: measured 1865.6 µs, predicted 924.2 µs
