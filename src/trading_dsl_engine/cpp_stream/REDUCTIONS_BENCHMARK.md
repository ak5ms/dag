# cpp_stream streaming-reduction benchmark

GitHub-hosted runner; one warmup and seven measured runs.

```text
rows=1,000,000 instruments=9 features=3 warmups=1 runs=7
full_median=11.146596 M rows/s seconds=0.089713 bytes=216000000
sum_axis0_median=24.574223 M rows/s seconds=0.040693 bytes=216
native_reduction_speedup=2.205x
full_plus_numpy_reduction_seconds=0.206440
native_vs_full_plus_post_speedup=5.073x
mean_axis0_median=24.647089 M rows/s
std_axis0_median=16.583695 M rows/s
cumsum_emit_last_median=94.695194 M rows/s
checksum=-106.013009915
```
