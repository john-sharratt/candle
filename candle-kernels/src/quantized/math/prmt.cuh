// The byte-permute helpers that build half2 pairs for the LOP3 dequantization
// of Q4_K and Q6_K.
//
// `prmt_pair_from_bytes` and `prmt_extract_byte` place bytes into 16-bit slots
// for LUT lookup. `prmt_build_lop3_pair` constructs the pair the LOP3
// dequantization consumes, and the `prmt_build_lop3_pair_0` to `_3` variants
// are compile-time constants.

// =============================================================================
// PRMT (BYTE PERMUTE) HELPER FUNCTIONS
// =============================================================================
// PRMT is a powerful PTX instruction that performs byte-level permutation
// on two 32-bit inputs. Each byte of the output is selected from one of 8
// possible source bytes (4 from each input).
//
// Selector nibble: 0-3 = bytes from 'a', 4-7 = bytes from 'b'
// Mode bits in selector byte [7:6]: 0=copy, 1=zero-fill if sign, 2=replicate sign
//
// Examples:
//   prmt(a, b, 0x7610) → [b[3], b[2], a[1], a[0]] (interleave)
//   prmt(a, 0, 0x1010) → [a[1], a[0], a[1], a[0]] (replicate low half)
//   prmt(a, b, 0x0040) → [a[0], 0, a[0], 0] (scatter bytes)
// =============================================================================

// Generic PRMT: output[i] = src_byte[selector_nibble_i]
// where src_byte[0..3] = bytes of 'a', src_byte[4..7] = bytes of 'b'
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
    return result;
}

// Build half2 pair: [a_byte[i], a_byte[j]] where each byte goes to a 16-bit slot
// Used for constructing (lo, hi) pairs for LOP3 dequantization
// Input: a = [b3, b2, b1, b0] (4 bytes)
// Output: [a_byte[hi_idx] << 16 | a_byte[lo_idx]]
// prmt_pair_from_bytes(a, 0, 2) → [a[2], a[0]] = [(hi>>16)&0xFF | ((lo>>16)&0xFF)<<16]
__device__ __forceinline__ uint32_t prmt_pair_from_bytes(uint32_t a, int lo_idx, int hi_idx) {
    // selector: byte 0 = lo_idx, byte 2 = hi_idx (bytes 1,3 are zero-filled via 0xF nibble)
    uint32_t selector = lo_idx | (0xF << 4) | (hi_idx << 8) | (0xF << 12);
    return prmt(a, 0, selector);
}

// Replicate half to half2: input = uint32_t containing half in low 16 bits
// Output: same half replicated to both slots
// This uses prmt to copy bytes [1,0] to [3,2,1,0]
__device__ __forceinline__ uint32_t prmt_replicate_lo_half(uint32_t h) {
    // h = [?, ?, h1, h0], output = [h1, h0, h1, h0]
    return prmt(h, h, 0x1010);
}

// Replicate high half to both slots
__device__ __forceinline__ uint32_t prmt_replicate_hi_half(uint32_t h) {
    // h = [h3, h2, ?, ?], output = [h3, h2, h3, h2]
    return prmt(h, h, 0x3232);
}

// Extract byte and place in low position of 16-bit slot (for LUT lookup)
// byte_idx: 0-3, specifies which byte to extract
__device__ __forceinline__ uint32_t prmt_extract_byte(uint32_t a, int byte_idx) {
    // Place byte in position 0, zero the rest
    return prmt(a, 0, byte_idx);
}

// =============================================================================
// PRMT PAIR CONSTRUCTION FOR Q4_K/Q6_K LOP3 DEQUANTIZATION
// =============================================================================
// These functions build (lo_byte, hi_byte) pairs needed for half2/bf162 LOP3 
// weight conversion. Given lo=[b3,b2,b1,b0] and hi=[B3,B2,B1,B0], we need:
//   pair0 = [B0, 0, b0, 0]  (bytes 0)
//   pair1 = [B1, 0, b1, 0]  (bytes 1)  
//   pair2 = [B2, 0, b2, 0]  (bytes 2)
//   pair3 = [B3, 0, b3, 0]  (bytes 3)
//
// PRMT selector for pair_i: we want lo_byte[i] at position 0, hi_byte[i] at position 2
// So selector = i | (0xF << 4) | ((4+i) << 8) | (0xF << 12)
// =============================================================================

// Build pair from corresponding bytes of lo and hi
// pair_idx: 0-3, output = [hi_byte[pair_idx], 0, lo_byte[pair_idx], 0]
__device__ __forceinline__ uint32_t prmt_build_lop3_pair(uint32_t lo, uint32_t hi, int pair_idx) {
    // lo bytes are at positions 0-3, hi bytes are at positions 4-7
    // We want: byte0=lo[pair_idx], byte1=0, byte2=hi[pair_idx], byte3=0
    uint32_t selector = pair_idx | (0xF << 4) | ((4 + pair_idx) << 8) | (0xF << 12);
    return prmt(lo, hi, selector);
}

// Compile-time constant versions for better optimization
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_0(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF4F0);  // [hi[0], 0, lo[0], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_1(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF5F1);  // [hi[1], 0, lo[1], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_2(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF6F2);  // [hi[2], 0, lo[2], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_3(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF7F3);  // [hi[3], 0, lo[3], 0]
}
