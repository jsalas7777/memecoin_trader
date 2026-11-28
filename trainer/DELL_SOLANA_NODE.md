# Dell computers that can run a Solana RPC node

Researched 2026-11-26. Prices are list prices or listings found online and change often
(server memory prices are especially volatile in 2026). Get a quote from Dell or the reseller before buying.

## What a Solana RPC node needs

| Part | Minimum | Recommended | Why |
|---|---|---|---|
| CPU | 16 cores / 32 threads | 24–32 cores, high clock (~3 GHz+), AMD Zen 3 or newer | keeps up with ~2,000 transactions/s |
| RAM | 256 GB ECC | **512 GB ECC** | needed for account indexes (`getProgramAccounts`, which gives the holder count) |
| Storage | 2 × 2 TB NVMe | enterprise NVMe, **accounts and ledger on separate drives**, plus snapshots | consumer SSDs wear out and stall under Solana's constant writes |
| Network | 1 Gbps symmetric | 10 Gbps symmetric, no data cap | tens of TB of traffic per month |
| OS / software | Ubuntu 22.04/24.04 | Agave or Firedancer as a non-voting RPC node, plus a Yellowstone gRPC (Geyser) plugin to stream pump.fun transactions | |

## The options

### 1. Best value: refurbished Dell PowerEdge R6525 (about $7,000–10,000)

| | |
|---|---|
| Listing found | R6525, 2 × AMD EPYC 7513 (32 cores each, 64 total), **512 GB DDR4**, 8 drive bays, no drives |
| Price | **$6,899.99** (pcserverandparts.com, marked down from $19,039.99) |
| Add | 3 enterprise NVMe drives (2–4 TB each, e.g. accounts, ledger, snapshots): roughly $1,000–2,500 (estimate) |
| **Total** | **~$8,000–9,500** |

- **Good:** already has the full 512 GB, and DDR4 is much cheaper than the DDR5 newer servers use. EPYC 7513 is Zen 3, which meets the CPU generation requirement.
- **Watch out:**
  - It's a 1U rack server: very loud, and it needs a rack or a data center.
  - It's dual-socket: two CPUs, so memory is split between them.
  - The 7513's base clock is 2.6 GHz, lower than ideal. EPYC 75F3 (2.95 GHz) versions are better for Solana if you find one.
  - Check that the drive bays and backplane support NVMe, not just SAS/SATA.
- **Similar:** R7525 (2U, more drive bays), e.g. 2 × EPYC 7763, 512 GB, 24 × 7.68 TB NVMe for about **$19,954** on Newegg. That's more storage than needed.

### 2. New server: Dell PowerEdge R7615 (single AMD EPYC 9004)

| | |
|---|---|
| Base price | **from $10,098.99** on Dell US (EPYC 9224, 16 GB, 1 HDD) |
| Configure as | one high-clock EPYC (e.g. a 32-core "F" model), 512 GB DDR5 RDIMM, 3+ NVMe drives |
| Expected total | **well above the base price**: 512 GB of DDR5 server memory alone is about $8,000–24,000 at mid-2026 module prices |

- **Good:** new, warranty, single CPU (simpler than dual), 2U with room for many NVMe drives.
- **Similar:** R6615 is the 1U version, from $14,998.99 on Dell US. Resellers (e.g. ServerMall) list both from about $7,500 base.
- For newer EPYC 9005 CPUs, Dell's single-socket models are, as far as I know, the R6715 and R7715. I couldn't confirm these on Dell's site, so ask Dell.

### 3. Quiet desk option: Dell Precision 7875 Tower (AMD Threadripper PRO)

| | |
|---|---|
| Base price | **from about $4,958–7,478** (Dell US, depending on configuration) |
| Memory | 8 slots, up to 2 TB DDR5 ECC RDIMM |
| Dell's 512 GB option | **+$41,215.39** (8 × 64 GB), so buying memory elsewhere is far cheaper |

- **Good:** a tower that's much quieter than a rack server and can sit in an office. High-clock Threadripper PRO CPUs suit Solana.
- **Watch out:** the memory upgrade cost; and a home internet line is usually not enough for a node anyway (see below).
- Not the Dell Pro Max Tower T2: it tops out at 128 GB.

## Before you buy: what it costs to run

| | |
|---|---|
| Internet | **the usual blocker**: 1–10 Gbps symmetric with no data cap. Home lines rarely qualify, so the machine usually goes in a data center (hosting a comparable server: about $1,500–3,000/month) |
| Electricity at home | roughly $50–100/month (estimate: ~500 W around the clock) |
| Your time | initial sync, upgrades every few weeks, fixing stalls when it falls behind |
| **History** | **a new node has no past transactions.** It starts from a recent snapshot and only records from the day it starts. For past coins you still need an archive (Old Faithful, ~250 TB) or a provider |

## Comparison

| Option | Upfront | Monthly |
|---|---|---|
| Refurbished R6525 (512 GB) + NVMe | ~$8,000–9,500 | power at home (if your internet can handle it) or ~$1,500–3,000 hosted |
| New R7615 configured for Solana | ~$20,000–35,000+ (mostly memory) | same |
| Precision 7875 + third-party memory | ~$13,000–30,000 | same |
| Rented dedicated node | $0 | ~$2,200–3,800 |
| Shared RPC plan (Helius, Chainstack) | $0 | **$0–200**, enough for training data |

**Recommendation:** if you want to own one, the **refurbished PowerEdge R6525 with 512 GB** is the best value by far.
For this project's training data, a $49/month RPC plan gets the same data with no hardware.
An own node makes sense once the live bot needs the lowest latency.

## Sources

- [PowerEdge R6615 (Dell US)](https://www.dell.com/en-us/shop/dell-poweredge-servers/poweredge-r6615-rack-server/spd/poweredge-r6615/pe_r6615_tm_vi_vp_sb)
- [PowerEdge R7615 (Dell US)](https://www.dell.com/en-us/shop/dell-poweredge-servers/poweredge-r7615-rack-server/spd/poweredge-r7615/pe_r7615_tm_vi_vp_sb)
- [PowerEdge R7615 (Dell US, configurator)](https://www.dell.com/en-us/shop/cty/pdp/spd/poweredge-r7615/pe_r7615_tm_vi_vp_sb)
- [PowerEdge R6615 (Dell Canada)](https://www.dell.com/en-ca/shop/ipovw/poweredge-r6615)
- [Dell PowerEdge R6615 (ServerMall)](https://servermall.com/en-us/sets/dell-poweredge-r6615-server/) · [R7615 (ServerMall)](https://servermall.com/en-us/sets/dell-poweredge-r7615-server/)
- [Refurbished Dell PowerEdge R6615 (NewServerLife)](https://newserverlife.com/server-models/dell-poweredge-r6615/)
- [Dell PowerEdge R6615 (Uvation)](https://marketplace.uvation.com/dell-poweredge-r6615-rack-server/)
- [PowerEdge R6525, 2 × EPYC 7513, 512 GB (pcserverandparts.com)](https://pcserverandparts.com/dell-poweredge-r6525-8-bay-sff-server-2x-epyc-7513-2-60ghz-32-core-128mb-cache-512gb-ddr4-h745-raid-no-hdd/)
- [PowerEdge R7525 listing (Newegg)](https://www.newegg.com/p/2NS-0008-7CC37)
- [PowerEdge R6525 / C6525 512 GB listings (Bytestock)](https://shop.bytestock.com/dell-poweredge-r6525-2-x-64-core-2-25ghz-amd-epyc-7742-512gb-6-x-6-4tb-nvme-ssd-h755n)
- [Precision 7875 Tower (Dell US)](https://www.dell.com/en-us/shop/desktop-computers/precision-7875-tower-workstation/spd/precision-t7875-workstation/xctopt7875mtus_vp)
- [Dell Precision 7875 review (StorageReview)](https://www.storagereview.com/review/dell-precision-7875-review-threadripper-pro-9995wx-meets-dual-rtx-pro-6000-blackwell-gpus)
- [Dell Precision towers explained (PCSP)](https://pcserverandparts.com/news/dell-precision-tower-workstations-explained-2026/)
- [Agave validator requirements (Anza)](https://docs.anza.xyz/operations/requirements)
- [Why Solana validators run on bare metal (Solana Foundation)](https://solana.com/news/why-solana-validators-run-on-bare-metal)
- [How much does it cost to self-host a node in 2026? (Chainstack)](https://chainstack.com/cost-to-self-host-blockchain-node-2026/)
- [Self-hosted Solana RPC node: hardware, cost, ops (Dysnix)](https://dysnix.com/blog/solana-rpc-node-infrastructure)
- [What Solana nodes actually demand in 2026 (Deeside)](https://www.deeside.com/the-hidden-tax-of-underpowered-infrastructure-what-solana-nodes-actually-demand-in-2026/)
- [Solana validator + Yellowstone Geyser setup (Subglow)](https://subglow.io/solana-validator-geyser-setup)
- [RDIMM memory prices, June 2026 (Fusion)](https://info.fusionww.com/blog/rdimm-shortages-and-the-shift-in-server-shipments?hsLang=en)
- [Old Faithful archive (Triton)](https://docs.triton.one/project-yellowstone/old-faithful-historical-archive)
