/* AUTO-GENERATED from config/tile.yaml by gen/gen_from_yaml.py — DO NOT EDIT. */
#ifndef NMGR_REGS_H
#define NMGR_REGS_H

/* micro-arch parameters (must match RTL by construction) */
#define NMGR_BANKS       4
#define NMGR_TILE_M      64
#define NMGR_K_PER_BANK  32
#define NMGR_X_BITS      8
#define NMGR_W_BITS      4
#define NMGR_ACC_BITS    32
#define NMGR_OUT_SHIFT   8
#define NMGR_OUT_BITS    8
#define NMGR_THRESHOLD   0
#define NMGR_CONTRACTION_K (NMGR_BANKS * NMGR_K_PER_BANK)

/* illustrative MMIO register map (word offsets) */
#define NMGR_REG_CTRL        0x00
#define NMGR_REG_STATUS      0x04
#define NMGR_REG_THRESHOLD   0x08
#define NMGR_REG_SURV_COUNT  0x0C
#define NMGR_REG_BITMAP_LO   0x10
#define NMGR_REG_BITMAP_HI   0x14

#endif /* NMGR_REGS_H */
