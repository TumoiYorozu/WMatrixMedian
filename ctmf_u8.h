#ifndef CTMF_U8_H
#define CTMF_U8_H

void ctmf_u8(
    const uint8_t *const src,
    uint8_t *const dst,
    const int width,
    const int height,
    const int src_step,
    const int dst_step,
    const int r,
    const int cn,
    int num_stripes);

#endif
