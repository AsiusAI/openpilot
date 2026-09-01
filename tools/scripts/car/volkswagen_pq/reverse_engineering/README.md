# SW 3201 reverse-engineering artifacts

Files in this directory:

- `1K0909144M_3201_golf_live_patched.bin`: read-only CCP upload of addresses
  `0x00000` through `0x5FFFF` from the live rack.
- `pq3201_decomp.c`: bulk Ghidra V850 little-endian decompiler export.
- `pq3201_disasm.txt`: bulk Ghidra disassembly export.
- `ExportAllDecomp.java` and `ExportAllDisasm.java`: Ghidra headless export
  scripts used to produce those artifacts.

Import parameters used for the raw image were V850 little-endian with image
base `0x00000000`. Application `GP` is `0x03FF9100`.

The captured image SHA-256 is
`45ddcabaed119c4e2e759ffbc4bee03dcf81c912a2d3ba55abce611708b71b8d`.
