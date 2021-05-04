#ifndef TAPA_SSSP_PIHEAP_INDEX_H_
#define TAPA_SSSP_PIHEAP_INDEX_H_

#include "sssp-kernel.h"

constexpr int kPifcSize = 4096;

using uint_qid_t = ap_uint<log2(kQueueCount)>;

using uint_pifc_tag_t =
    ap_uint<kVidWidth - log2(kPifcSize) - log2(kQueueCount)>;
using uint_pifc_index_t = ap_uint<log2(kPifcSize)>;

struct HeapIndexCacheEntry {
  bool is_dirty;
  uint_pifc_tag_t tag;
  HeapIndexEntry index;

  void UpdateTag(uint_vid_t vid) {
    this->tag.range() = vid.range(kTagMsb, kTagLsb);
  }

  Vid GetVid(uint_pifc_index_t index, uint_qid_t qid) const {
    return tag, index, qid;  // Concatenation.
  }

  bool IsHit(uint_vid_t vid) const {
    return this->tag == vid.range(kTagMsb, kTagLsb);
  }

  static constexpr int kQidLsb = 0;
  static constexpr int kQidMsb = kQidLsb + log2(kQueueCount) - 1;
  static constexpr int kIndexLsb = kQidMsb + 1;
  static constexpr int kIndexMsb = kIndexLsb + log2(kPifcSize) - 1;
  static constexpr int kTagLsb = kIndexMsb + 1;
  static constexpr int kTagMsb = kVidWidth - 1;
};

inline uint_pifc_index_t GetPifcIndex(uint_vid_t vid) {
  return vid.range(HeapIndexCacheEntry::kIndexMsb,
                   HeapIndexCacheEntry::kIndexLsb);
}

#endif  // TAPA_SSSP_PIHEAP_INDEX_H_
