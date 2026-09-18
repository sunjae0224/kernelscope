# SM89_RTX4090 model derivation (2026-09-18)

This is an **uncalibrated Ada resource model using Ampere timing and HMMA**.
Passing vector-add replay is a functional check, not attention timing validation.
The canonical files are [SM89_RTX4090/](SM89_RTX4090/); install with
`bash env/setup_accelsim_4090.sh config`. No `.icnt` file is needed: both parent
and derived models use the built-in crossbar (`network_mode=2`).

Sources:

- **host**: CUDA device query from `vecadd_4090.cu`, saved in
  `/home/skkai/accelsim/logs/vecadd-native.log`.
- **Ada**: [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html),
  occupancy, shared memory, and binary compatibility sections.
- **whitepaper**: [NVIDIA Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).
- **parent**: GPGPU-Sim commit `91880c53383d5a6a6742bfb1be2c5f34e39c7871`,
  `configs/tested-cfgs/SM86_RTX3070/gpgpusim.config`.
- **code**: the same commit's `src/gpgpu-sim/{gpu-cache.cc,hashing.cc,addrdec.cc,dram.cc}`.

| Knob | SM86_RTX3070 | SM89_RTX4090 | Source / rationale | Confidence |
|---|---|---|---|---|
| `gpgpu_n_clusters` | 46 | 128 | host SM count | high |
| `gpgpu_n_cores_per_cluster` | 1 | 1 | retain one SM per cluster abstraction | high |
| `gpgpu_clock_domains` | 1132:1132:1132:3500.5 | 2520:2520:2520:5250 | whitepaper nominal boost; DRAM 21 Gbps / ratio 4; icnt/L2 tied to core by assumption | med (icnt/L2 low) |
| `gpgpu_n_mem` | 16 | 24 | parent uses 16-bit subchannels, not 8 x 32-bit channels; 24 x 16 = 384 bits | high aggregate / med topology |
| `gpgpu_n_sub_partition_per_mchannel` | 2 | 2 | 48 L2 slices in derived model, inherited topology abstraction | med |
| `gpgpu_dram_buswidth` | 2 bytes | 2 bytes | parent/code | med |
| `dram_data_command_freq_ratio` | 4 | 4 | parent/code; 24 x 2 x 4 x 5250 MHz = 1008 GB/s | med |
| `gpgpu_dram_burst_length` | 16 | 16 | inherited GDDR model | low |
| `gpgpu_cache:dl2` geometry | 64 sets x 128 B x 16 ways | 1024 sets x 128 B x 12 ways | 48 slices x 1024 x 128 x 12 = 75,497,472 B; sets/ways chosen for representable capacity, not measured associativity | high capacity / low geometry |
| `gpgpu_cache:dl2` set index | P | X | IPoly supports only 16/32/64 sets and aborts for 1024; XOR supports this model and all L2 variants | low hardware fidelity |
| `gpgpu_memory_partition_indexing` | 2 | 2 | inherited IPoly with modulo for non-power-of-two channel count; code supports 24 channels | low hardware fidelity |
| `gpgpu_shader_registers` / `gpgpu_registers_per_block` | 65536 / 65536 | same | host/Ada | high |
| `gpgpu_shader_core_pipeline` | 1536:32 | same | host/Ada, 48 resident warps | high |
| `gpgpu_shader_cta` | 32 | 24 | Ada limit; parent file says 32 even though GA10x hardware allows 16 | high |
| `gpgpu_shmem_size` / `gpgpu_shmem_sizeDefault` | 102400 / 102400 B | same | host/Ada | high |
| `gpgpu_shmem_per_block` | 49152 | 101376 | host/Ada opt-in block limit (99 KiB) | high |
| `gpgpu_shmem_option` | 0,8,16,32,64,100 | same | Ada carveouts | high |
| `gpgpu_unified_l1d_size` | 128 KiB | same | Ada | high |
| `gpgpu_ptx_force_max_capability` | 86 | 89 | host; trace replay remains limited to mapped opcodes | high |
| `gpgpu_compute_capability_major/minor` | 8/6 | 8/9 | host | high |
| `gpgpu_occupancy_sm_number` | 86 | 89 | CUDA occupancy target | high |
| `gpgpu_coalesce_arch` | 86 | 89 | both use modern sector coalescing path in code | med |
| `gpgpu_num_sched_per_core` / subcore model | 4 / 1 | same | parent/Ada | high |
| `gpgpu_tensor_core_avail` | 1 | **1, unchanged** | required; [issue 451](https://github.com/accel-sim/accel-sim-framework/issues/451) | med for common HMMA subset |
| trace HMMA latency/initiation | 32/32 | same | parent `trace.config`, not Ada-calibrated | low |
| `gpgpu_l2_rop_latency` / `dram_latency` | 187 / 254 | same | inherited, not measured on Ada | low |
| DRAM timing string | parent timing string | unchanged | inherited GDDR6 approximation to GDDR6X | low |
| all other options | parent values | unchanged | no measured basis for changing them | low timing / med structure |

The task's assumed parent values (8 controllers, 16 CTAs) differ from the actual
pinned file (16 controllers, 32 CTAs). The derivation uses the checked-out file.
L2 is the enabled **72 MiB** on this RTX 4090, not full AD102's 96 MiB.
The 3105 MHz `nvidia-smi` maximum is not the chosen operating point. Convert
simulated cycles with **sim_us = cycles / 2520** and pass `--clock-mhz 2520` to
`kernelscope report` (its shared default remains A100's 1410 MHz).

Native sm_89 SASS is rejected by unmodified Accel-Sim v2.0.0 with
`unsupported binary version: 89`, even though the process returns zero.
[accelsim-sm89.patch](accelsim-sm89.patch) routes version 89 to the existing
Ampere opcode map; it does not implement Ada-only instructions. Unknown opcodes
still fail. The unpatched toolchain gate used sm_86 SASS executed on this 4090.
Keep this distinction when reporting architecture support.

All seven what-if variants have CPU checks for exactly one changed option.
The initial L2 IPoly configuration failed before replay; the final XOR version
replayed vector-add to a clean exit at **18,939 cycles** (sm_86 trace).
Logs: `/home/skkai/accelsim/logs/vecadd-sm89-{ipoly-failed,model}.log`.
