#ifndef CTMF_U16_H
#define CTMF_U16_H

void ctmf_u16(
    const uint16_t *const src,
    uint16_t *const dst,
    const int width,
    const int height,
    const int src_step,
    const int dst_step,
    const int r,
    const int cn,
    int num_stripes);

#endif
