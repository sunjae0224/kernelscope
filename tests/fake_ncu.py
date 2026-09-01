"""Stand-in for the `ncu` binary: records its argv, runs the target, prints a canned CSV on stdout.

Usage (as the sweep would call ncu):
    python tests/fake_ncu.py --csv --metrics ... -k regex:... --launch-skip N --launch-count M <target...>
"""
import json
import os
import subprocess
import sys

CSV = '''"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","python","127.0.0.1","fake_kernel(int)","1","7","(128, 1, 1)","(32, 1, 1)","0","8.0","Command line profiler metrics","gpu__time_duration.sum","usecond","12.5"
"0","1","python","127.0.0.1","fake_kernel(int)","1","7","(128, 1, 1)","(32, 1, 1)","0","8.0","Command line profiler metrics","launch__grid_size","","32"
'''


def main():
    argv = sys.argv[1:]
    # everything after --launch-count <n> is the target command
    i = argv.index("--launch-count")
    target = argv[i + 2:]
    if os.environ.get("FAKE_NCU_ARGS_FILE"):
        with open(os.environ["FAKE_NCU_ARGS_FILE"], "w") as f:
            json.dump(argv[:i + 2], f)
    proc = subprocess.run(target, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    if os.environ.get("FAKE_NCU_FAIL"):
        # real ncu reports driver/permission problems on STDOUT and exits 9
        sys.stdout.write("==PROF== Connected to process 1\n==ERROR== Profiling failed because a driver resource was unavailable.\n")
        sys.exit(9)
    sys.stdout.write("==PROF== Connected to process 1\n")
    sys.stdout.write(CSV)
    sys.stdout.write("==PROF== Disconnected from process 1\n")


if __name__ == "__main__":
    main()
