#ifndef SVT_AV2_PYRAMID8_H
#define SVT_AV2_PYRAMID8_H

/* Interior pictures are coded in preorder. Each level has its own DPB slot,
 * reusable after its left subtree has finished. Even interior anchors leave
 * exactly one hidden display to append after each odd leaf picture. */
typedef struct SvtAv2PyramidNode {
    unsigned left, right, position;
    unsigned left_slot, right_slot, slot;
} SvtAv2PyramidNode;

static inline SvtAv2PyramidNode svt_av2_pyramid_node(unsigned span, unsigned index, unsigned end_slot) {
    SvtAv2PyramidNode n = {0,span,span,end_slot^1u,end_slot,end_slot};
    for (unsigned depth=2u; index; ++depth) {
        const unsigned length = n.right-n.left;
        n.position = n.left + (length==2u ? 1u : 2u*((length+2u)/4u));
        n.slot = depth;
        if (--index == 0u) break;
        const unsigned left_count = n.position-n.left-1u;
        if (index <= left_count) {
            n.right = n.position;
            n.right_slot = n.slot;
        } else {
            index -= left_count;
            n.left = n.position;
            n.left_slot = n.slot;
        }
    }
    return n;
}

static inline unsigned svt_av2_pyramid_offset(unsigned span, unsigned index) {
    return svt_av2_pyramid_node(span,index,0u).position;
}

/* Full groups alternate their end anchors between 0 and 1. The midpoint
 * lives in slot 2, while the two disjoint quarter points share slot 3. */
static inline unsigned svt_av2_pyramid8_slot(unsigned display) {
    if (!(display & 7u)) return (display >> 3u) & 1u;
    return (display & 3u) ? 3u : 2u;
}
#endif
