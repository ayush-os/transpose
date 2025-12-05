# transpose

(main) root@C.28510276:/workspace/transpose/build$ ./bin/transpose 
--- Matrix Transpose Stable Timing Test (N x N) ---
Matrix Dimension N: 4096 x 4096
Total Array Size: 16777216 elements (0.0625 GB)
Grid: 256x256 blocks, 16x16 threads/block.
Warming up the GPU and Caches (10 runs)...
Starting Stable Timing Loop (1000 runs)...

--- Timing Results ---
Total execution time for 1000 stable runs: 161.143 ms
**Average kernel execution time:** 161.143 us

Verification Check: **PASSED**
  A[1234][4032] (5.0585e+06) correctly moved to B[4032][1234]

each reads 2 floats * 16,777,216 elems * 4 bytes/float = 134,217,728 bytes read

134,217,728 bytes / 0.000161143 s = 832,910,694,228.107953805 B/s ~ 833Billion B/s

which is 833billion /  1.55 TB about 0.5374193548 ~ 53.7% bandwidth utilization

optimization 1: global memory coalescing on writes as opposed to on reads

if you coalesce on writes then its better than coalescing on reads cuz of cache write-back which is more expensive than just L2 cache reads in the case of reads

this took my kernel from 161.143 us to 143.597 us, which is an 11% decrease! No small amount

this also increased bandwidth utilization to 134,217,728 bytes / 0.000143597 = 934,683,370,822.5102195728 B/s ~ 934.68 B/s

which is 934.68 bill / 1.55 trill = 0.6030193548 => 60.3% utilization which is a 12.3% increase in bandwidth utilization