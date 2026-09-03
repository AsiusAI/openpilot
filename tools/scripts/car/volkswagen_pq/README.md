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

## Live validation

This sequence was completed successfully on the Golf's `1K0909144M` SW `3201`
rack. The accepted permanent request was:

```text
31 BB 01 03 00 01 08 38 00 1C 2D 2D
```

After a full ignition power cycle, Channel 6 read back as `1` and the rack's
`LH2_Sta_HCA` state changed from `0` (`DISABLED`) to `3` (`READY`). A subsequent
road test engaged and disengaged normally, produced nonzero steering torque,
and completed without a temporary or permanent EPS steering fault.

These observations validate the HCA adaptation procedure for this exact rack
and software only. They do not establish compatibility with another rack,
software version, ABS module, or longitudinal-control configuration.

The live disabled response was
`71 BA 01 03 82 03 00 00 04 25 00 88 FF`; the two-byte Channel 6 value is
`00 00`. The trailing `00 88` is metadata rather than a channel value or echo.

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

This only enables lateral HCA reception in the rack. The ABS is not involved in
accepting steering torque.

## ABS and experimental longitudinal control

Read-only diagnostics identified the Golf's brake controller as:

```text
part:      1K0 907 379 BJ
software:  0121
component: ESP MK60EC1 (H31)
coding:    143B400D112800FB281402E7881F0040350000
```

No ABS coding, adaptation, actuator test, or brake request was sent while
collecting this identity. The
[comma openpilot Volkswagen-PQ compatibility table](https://github.com/commaai/openpilot/wiki/Volkswagen-PQ)
lists `1K0 907 379 BJ/BL` H31 as ACC-capable with Follow-to-Stop, but not full
Stop-and-Go. This establishes a plausible hardware path; it does not establish
that this particular car is already coded for ACC or that radarless openpilot
longitudinal control is safe to use.

For this exact BJ/H31 family, the
[PQ35 MK60EC1 coding notes](https://wiki.pq35.de/Electronics/03-ABS_Brakes/MK60EC1)
and a [published BJ/H31 retrofit record](https://de.scribd.com/document/692779689/Umruestung-Acc-Final)
independently identify the ACC selection
as ABS long-coding Byte 16 Bit 5: `1` means ACC is not installed, and clearing
it selects ACC. Never copy a complete long-coding string from another car:
MK60EC1 coding includes VIN, brake, drivetrain, and equipment data.
The live read returned 19 coding bytes. Byte 16 is `0x35`, so Bit 5 is `1` and
the controller is currently coded with ACC not installed. Before any possible
write, preserve this exact original value, verify
that Byte 16 Bit 5 is the only intended delta, and prepare the exact original
value as the rollback.

The utility has a read-only ABS action for that prerequisite:

```bash
python tools/scripts/car/vw_pq_config.py abs-info
```

It is limited to `1K0907379BJ` SW `0121`, displays the complete coding and the
decoded ACC bit, and contains no ABS write operation. Its TP2.0 receiver handles
all four data opcodes and the 15-bit length field used by MK60EC1; the earlier
receiver interpreted the application flag as part of the length and could
misassociate a later response.

The current radarless Golf integration must also not be assumed ready for road
longitudinal testing. With no stock ACC radar message to copy, `CarState.acc_type`
currently remains `0` (`Basis_ACC`), while this ABS variant is specifically a
Follow-to-Stop controller and the PQ DBC defines type `1` for that mode. Stock
AEB is not preserved by the PQ openpilot-long path. These must be resolved and
bench/non-actuating validation completed before enabling experimental long.

The engine controller was also identified read-only:

```text
part:      03C 906 016 AJ
software:  9458
component: MED17.5.5 G (CAXA 1.4 TSI)
coding:    00004D (short coding)
```

Factory retrofit procedures change the engine ECU from conventional cruise
(GRA) to ACC as well as changing the ABS bit. A successful test on the same
`03C906016AJ` ECU family established that this is a KWP2000 Access Authorization
Code 2 (VCDS Coding II/Login 2) configuration, not an engine firmware patch.
The reversible radarless test sequence is `16167` (disable cruise/ACC), followed
by `13377` (enable ACC without Follow-to-Stop). The stock rollback is `16167`,
then `11463` (enable conventional cruise), followed by an ignition cycle.

Community references also list `30903` for automatic transmissions with
Follow-to-Stop and Front Assist, but that mode is deliberately excluded here:
this car has no radar/Front Assist and its H31 ABS cannot safely hold brake
pressure at a stop. Automatic transmission alone is not a reason to select the
Follow-to-Stop configuration for this experiment.

```bash
python tools/scripts/car/vw_pq_config.py engine-info
```

`engine-info`, like `abs-info`, is read-only and prints identity plus the
controller's current short or long coding when available.

The guarded longitudinal configuration actions are:

```bash
# These are previews unless --experimental-long-write is also supplied.
python tools/scripts/car/vw_pq_config.py engine-acc-enable
python tools/scripts/car/vw_pq_config.py engine-stock-restore
python tools/scripts/car/vw_pq_config.py abs-acc-enable
python tools/scripts/car/vw_pq_config.py abs-stock-restore
```

They are hard-locked to the exact live engine and ABS identities above. The ABS
action accepts only the saved 19-byte pair and verifies that Byte 16 Bit 5 is
the sole delta (`0x35` to `0x15`, or the reverse). The engine action always
disables the existing mode before selecting ACC or stock cruise, and attempts
the stock-cruise activation immediately if ACC activation is rejected.

Even with the explicit write flag, each action prints every application-layer
request and requires a target-specific typed confirmation. The Code 2 request
uses KWP2000 service `27 02` plus the 16-bit code; ABS coding uses the VW
workshop fingerprint DID `F198` followed by coding DID `0600`. These wire
formats have unit coverage and public protocol support, but have not yet been
captured from ODIS on this exact ECU/ABS pair. Compare them against an ODIS or
VCDS trace before the first live longitudinal configuration write.

### Required validation order

1. With experimental longitudinal disabled, run `abs-info` and `engine-info`.
   Save the complete output and independently decode the original values.
2. Do not write until the ABS Byte 16 delta, engine Code 2 requests, rollback
   requests, battery conditions, diagnostic sessions, and raw ODIS/VCDS request
   framing are all confirmed. A gateway installation-list change is not
   justified for radarless operation unless logs prove a controller explicitly
   requires Address 13.
3. If a coding write is later approved, perform one controller at a time with
   engine off and ignition on, then power-cycle and read back before proceeding.
   Clear no DTCs until they have been recorded and understood.
4. Keep openpilot longitudinal disabled and observe normal manual braking,
   ABS/ESC warning lamps, stock cruise, engine response, and CAN status first.
   No actuator test or nonzero acceleration request belongs in this stage.
5. Only after the integration selects `ACS_Typ_ACC=1`, handles controller
   faults/status correctly, and explicitly addresses loss of stock AEB should
   a non-actuating zero/inactive-message test be considered. Any later motion
   test must start in a controlled area with a driver ready on the brake and
   must validate positive acceleration separately from commanded deceleration.

Follow-to-Stop is not Stop-and-Go: the H31 compatibility result does not imply
automatic restart from a standstill. Radarless/vision-only ACC on this Golf is
an experimental integration, not the documented OEM radar architecture.

Later H31 testing reported a more specific standstill limitation: MK60EC1 can
brake to zero but drops brake pressure rather than holding it indefinitely when
stopped through the TSK/`ACS_Anhaltewunsch` path. The engine then requests
braking again, potentially creating a release/re-brake cycle. Treat H31 as
mandatory driver brake takeover at the stop. Initial openpilot-long support
must disengage or hand over before standstill and must not advertise auto-hold,
Stop-and-Go, or auto-resume. The moving-speed controller can be evaluated
separately only after the ABS and engine prerequisites are satisfied.
