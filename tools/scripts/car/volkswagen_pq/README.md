# Volkswagen PQ steering-rack diagnostics

This directory records the investigation of Heading Control Assist (HCA) on a
Golf Mk6/PQ35 with steering rack `1K0909144M`, software `3201`.

## Finding

The rack was receiving normal vehicle CAN traffic but advertised
`LH2_Sta_HCA = 0` (`DISABLED`) in `Lenkhilfe_2` (`0x3D2`). This explains why
openpilot could identify the car, read steering angle, and engage in software,
while the rack ignored HCA torque commands.

Address 44 adaptation Channel 6 must be set from `0` to `1`. This rack does not
use a generic KWP `WriteDataByLocalIdentifier` (`3B`) request for the change.
Both the SW 3201 firmware and the independently recovered Carista request
builders use local routine `0x0103` with subfunctions `B8`, `B9`, `BA`, and
`BB`. The SW 3201 handler does not gate this routine on KWP SecurityAccess
(`27`), so an OBDEleven PIN cannot fix a client using the wrong operation or
sequence.

The sequence is:

```text
10 89                         diagnostic session
1A 9B                         ECU identity and current six-byte workshop code
31 B8 01 03                   start short-adaptation routine
31 BA 01 03                   pre-read; response payload 81
31 B9 01 03 06                select Channel 6
31 BA 01 03                   read Channel 6
31 B9 01 03 00 01             set value 1 temporarily
31 BA 01 03                   verify temporary value
31 BB 01 03 00 01 <WSC x6>    write value 1 permanently
32 B8 01 03                   stop routine
```

The permanent request format is proven in rack function `0x0002f66c`: `BB`
passes the two-byte channel value and then exactly six workshop-code bytes to
the nonvolatile-write path at `0x0002f284`. A persistent write may respond with
KWP NRC `0x78` while it completes.

Run the guarded utility from the openpilot root after stopping the comma
processes:

```bash
python tools/scripts/car/vw_pq_config.py show
python tools/scripts/car/vw_pq_config.py enable
```

`show` performs no persistent write. `enable` is hard-limited to rack
`1K0909144M` SW `3201`, checks voltage, displays the exact temporary and
permanent requests, and requires the explicit phrase `ENGINE OFF, IGNITION ON`
before it sends either write. Use bus 1 for the Golf gateway harness used in
this investigation.

## Captured firmware

`reverse_engineering/1K0909144M_3201_golf_live_patched.bin` is the complete
`0x00000`-`0x5FFFF` CCP upload captured read-only from the live rack after its
low-speed/time-limit calibration patch was flashed.

```text
size:   393216 bytes
sha256: 45ddcabaed119c4e2e759ffbc4bee03dcf81c912a2d3ba55abce611708b71b8d
```

The Ghidra exports are included for reproducibility. The application uses
`GP = 0x03FF9100`; the main diagnostic dispatcher is `0x00023D58`, the SID
`31` dispatcher is `0x0002F15C`, and the short-adaptation handlers are
`0x0002F66C` and `0x0002F284`.

This only enables lateral HCA reception in the rack. A separate HCA-related ABS
change was reported by another retrofit attempt, but the Golf's exact ABS part,
software, and coding have not yet been captured or verified. Do not write the
ABS based on this rack procedure. ABS capability for longitudinal ACC braking
is also a separate issue.
