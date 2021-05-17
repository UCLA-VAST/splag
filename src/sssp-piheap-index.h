#ifndef TAPA_SSSP_PIHEAP_INDEX_H_
#define TAPA_SSSP_PIHEAP_INDEX_H_

#include "sssp-kernel.h"
#include "sssp-piheap.h"

constexpr int kPifcSize = 4096;

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

constexpr int kPiscSize = kLevelCount;
using uint_pisc_pos_t = ap_uint<bit_length(kPiscSize - 1)>;

struct HeapStaleIndexEntry : public HeapIndexEntry {
  Vid vid;
  bool matches(Vid vid) const {
    constexpr int kLsb = log(kQueueCount, 2);
    constexpr int kMsb = kVidWidth - 1;
    const bool result =
        valid() && ap_uint<kVidWidth>(this->vid).range(kMsb, kLsb) ==
                       ap_uint<kVidWidth>(vid).range(kMsb, kLsb);
    if (result) {
      CHECK_EQ(this->vid, vid);
    }
    return result;
  }
  using HeapIndexEntry::operator=;
  void set(Vid vid, const HeapIndexEntry& entry) {
    HeapIndexEntry::operator=(entry);
    this->vid = vid;
  }
};

template <int N>
inline void FindStaleIndex(Vid vid, const HeapStaleIndexEntry (&array)[N],
                           bool& is_match_found, uint_pisc_pos_t& match_idx,
                           bool& is_empty_found, uint_pisc_pos_t& empty_idx) {
  bool is_match_found_0, is_match_found_1;
  uint_pisc_pos_t match_idx_0, match_idx_1;
  bool is_empty_found_0, is_empty_found_1;
  uint_pisc_pos_t empty_idx_0, empty_idx_1;
  FindStaleIndex(vid, (const HeapStaleIndexEntry(&)[N / 2])(array),
                 is_match_found_0, match_idx_0, is_empty_found_0, empty_idx_0);
  FindStaleIndex(vid, (const HeapStaleIndexEntry(&)[N - N / 2])(array[N / 2]),
                 is_match_found_1, match_idx_1, is_empty_found_1, empty_idx_1);
  is_match_found = is_match_found_0 || is_match_found_1;
  match_idx =
      is_match_found_0 ? match_idx_0 : uint_pisc_pos_t(match_idx_1 + N / 2);
  is_empty_found = is_empty_found_0 || is_empty_found_1;
  empty_idx =
      is_empty_found_0 ? empty_idx_0 : uint_pisc_pos_t(empty_idx_1 + N / 2);
}

template <>
inline void FindStaleIndex<1>(Vid vid, const HeapStaleIndexEntry (&array)[1],
                              bool& is_match_found, uint_pisc_pos_t& match_idx,
                              bool& is_empty_found,
                              uint_pisc_pos_t& empty_idx) {
  is_match_found = array[0].matches(vid);
  match_idx = 0;
  is_empty_found = !array[0].valid();
  empty_idx = 0;
}

inline HeapIndexEntry GetStaleIndexLocked(
    const HeapStaleIndexEntry (&array)[kPiscSize], Vid vid, bool& is_pos_valid,
    uint_pisc_pos_t& pos) {
  // If found, pos is set to the entry, entry is returned; otherwise, pos is
  // set to an available (invalid) location.
  bool is_match_found;
  uint_pisc_pos_t match_idx;
  bool is_empty_found;
  uint_pisc_pos_t empty_idx;
  FindStaleIndex(vid, array, is_match_found, match_idx, is_empty_found,
                 empty_idx);
  is_pos_valid = is_match_found || is_empty_found;
  pos = is_match_found ? match_idx : empty_idx;
  return array[pos];
}

#endif  // TAPA_SSSP_PIHEAP_INDEX_H_
