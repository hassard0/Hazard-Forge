# NW1 — deterministic rollback netcode over REAL UDP (two-process demo)

Slice **NW1** adds a **real UDP transport** (`engine/net/udp_transport.h`, `hf::net::udp`) behind the same
interface shape as the shipped, socket-free rollback substrate (`engine/net/session.h` NS1–6 + the
`SimChannel` model in `engine/net/transport.h`). The whole point: the deterministic rollback netcode now
runs over **real datagrams** — the "deterministic rollback over a real network" claim UE5's float
architecture structurally cannot make.

## The honest boundary (🔵 nondeterministic I/O)

- **The simulation is deterministic**: inputs → identical world (bit-exact integer physics + the NS1–6
  lockstep/rollback). Two peers fed the same confirmed inputs re-derive a bit-identical world.
- **The transport is nondeterministic I/O**: real UDP may reorder / drop / delay datagrams, and the exact
  arrival order varies run-to-run. This is the **one** documented nondeterministic-I/O exception in the
  codebase.
- **The outcome is bit-exact**: the rollback layer (predict → snapshot → roll back on the corrected input,
  NS3–6) already tolerates arbitrary arrival order + loss-with-resend, so **both peers converge to the same
  pinned digest regardless of transport timing**.

So: *nondeterministic network, bit-exact deterministic outcome.*

## How it's gated (non-flaky)

`tests/udp_transport_test.cpp` opens **real UDP sockets** (Winsock on Windows, BSD sockets elsewhere) bound
to `127.0.0.1:0` (OS-assigned ports) **in one process** and exchanges real datagrams between two endpoints.
This exercises the real `sendto`/`recvfrom` path without flaky cross-process orchestration. A bounded
**ACK + resend** reliability loop guarantees eventual delivery (a drop-then-resend converges), so the
**digest assertions** — the pinned deterministic part — never flake. The test also injects
reorder / duplicate / drop-then-resend faults and asserts both peers **still** converge to the identical
pinned digest.

Pinned outcomes:

| proof | digest |
|---|---|
| GAME1 best-of-3 match over real UDP (both peers) | `0x78123003c3a55a37` |
| per-tick predict+rollback over real UDP (both peers) | `0x1aa9738bcc0c7001` |

Both equal the in-process / scripted values (`duel_test` / `session_test`) — real transport does **not**
change the outcome.

## The true two-process run (best-effort; localhost)

The loopback gate already drives real sockets. A genuine **two-OS-process** run over localhost is a thin
harness on top of the same `hf::net::udp::UdpTransport`: each process opens its own socket, they trade
ports (via a known port, a small rendezvous file, or argv), then run their half of the exchange. Sketch:

```
# terminal A — peer 0 binds a socket, prints its port, sends player-0 inputs, receives player-1 inputs
udp_duel_peer --self 0 --listen 40000 --peer 127.0.0.1:40001

# terminal B — peer 1 (mirror)
udp_duel_peer --self 1 --listen 40001 --peer 127.0.0.1:40000
```

Each process, after the exchange, prints its independently-reconstructed match digest; both print
`0x78123003c3a55a37`. Because real cross-process timing (scheduling, port races, firewall prompts) is
environment-dependent and **can't** be a reliable CI gate, this two-process form is **documented here** and
the **same-process loopback** is the committed gate. On a locked-down / headless box the two-process run may
be blocked by the OS firewall or unavailable ports; if so, the loopback test is the authoritative proof and
this document records the equivalence (both use the identical `UdpTransport` send/recv path — only the
process boundary differs, and the rollback layer is indifferent to it).

## Why this is a moat UE5 can't cross

Deterministic rollback over a real network requires a **bit-exact** simulation whose state can be
snapshotted, rolled back, and re-simulated to the identical result on every peer. UE5's Chaos physics is
**float** and **non-deterministic** across machines/compilers, so its networking is authoritative-server +
interpolation, **not** peer-symmetric rollback — it cannot guarantee two peers re-derive an identical world
from inputs alone. Hazard Forge's integer sim does, and NW1 shows it converging over **real UDP datagrams**,
not just an in-process model.
