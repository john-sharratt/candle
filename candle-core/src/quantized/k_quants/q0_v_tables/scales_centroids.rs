//! The scale and centroid tables, and the peak indices, that the Q0_V codec indexes
//! on each side.
//!
//! Each centroid table has 16 entries per scale row, because the kernel reads
//! `[16]` rows and the types must match. The K and V centroid tables are built by
//! duplicating the calibrated 8-entry table into adjacent slot pairs, so entry j
//! becomes 2j and 2j+1. A full 16-entry calibration has not been measured.

#[rustfmt::skip]
    pub const SCALE_TABLE_BITS_K: [u16; 32] = [
    
        0x01D0,0x0F3A,0x10B7,0x11BB,0x12BE,0x13B5,0x144C,0x14B4,
        0x1514,0x1570,0x15C8,0x161C,0x1670,0x16C4,0x1719,0x1770,
        0x17CB,0x1815,0x1846,0x187B,0x18B3,0x18F1,0x1935,0x1981,
        0x19D7,0x1A3B,0x1AB3,0x1B49,0x1C0D,0x1CAE,0x1DB6,0x2008
    
    ];

#[rustfmt::skip]
    pub(super) const CENTROID_TABLE_BITS_K_OLD: [[u16; 8]; 32] = [

      /* scale_idx =  0  (8253 src blocks) */ [ 0xA86D,0x9D30,0x99F0,0x9380,0x1400,0x1980,0x1CD0,0x26B2 ],
      /* scale_idx =  1  (44790 src blocks) */ [ 0xACA1,0xA2E4,0x9FE0,0x9B20,0x11C0,0x1D70,0x21D0,0x2CE8 ],
      /* scale_idx =  2  (37123 src blocks) */ [ 0xAE1E,0xA452,0xA114,0x9C40,0x1680,0x1FF0,0x23D0,0x2D1F ],
      /* scale_idx =  3  (35842 src blocks) */ [ 0xAF54,0xA4F6,0xA180,0x9C20,0x1930,0x20F8,0x24B0,0x2FB0 ],
      /* scale_idx =  4  (36114 src blocks) */ [ 0xB0C9,0xA5DC,0xA248,0x9C58,0x1AC0,0x21D0,0x25A6,0x3012 ],
      /* scale_idx =  5  (35808 src blocks) */ [ 0xB258,0xA698,0xA2DC,0x9C70,0x1C68,0x22E4,0x26A2,0x3028 ],
      /* scale_idx =  6  (36158 src blocks) */ [ 0xB152,0xA780,0xA3A4,0x9CA0,0x1D70,0x23E0,0x2796,0x3050 ],
      /* scale_idx =  7  (36233 src blocks) */ [ 0xB167,0xA830,0xA43A,0x9D10,0x1DC8,0x2464,0x2828,0x33A3 ],
      /* scale_idx =  8  (36056 src blocks) */ [ 0xB194,0xA874,0xA494,0x9DA0,0x1E70,0x24BA,0x2879,0x3283 ],
      /* scale_idx =  9  (36005 src blocks) */ [ 0xB15D,0xA8D2,0xA4EC,0x9DE8,0x1E98,0x2516,0x28D3,0x31D8 ],
      /* scale_idx = 10  (36228 src blocks) */ [ 0xB1F8,0xA90E,0xA544,0x9E80,0x1F18,0x2554,0x2910,0x31F4 ],
      /* scale_idx = 11  (36307 src blocks) */ [ 0xB220,0xA96C,0xA5AE,0x9F38,0x1F30,0x259E,0x295E,0x3220 ],
      /* scale_idx = 12  (35970 src blocks) */ [ 0xB2F0,0xA9B8,0xA610,0x9FD0,0x1FB8,0x2604,0x299D,0x3281 ],
      /* scale_idx = 13  (36282 src blocks) */ [ 0xB2EA,0xA9F4,0xA66A,0x9FF0,0x1FF0,0x2634,0x29D9,0x334A ],
      /* scale_idx = 14  (36018 src blocks) */ [ 0xB2CF,0xAA13,0xA682,0xA03C,0x200C,0x2652,0x2A07,0x333F ],
      /* scale_idx = 15  (35878 src blocks) */ [ 0xB441,0xAA6D,0xA6FE,0xA0A8,0x1FE0,0x2680,0x2A2A,0x33A0 ],
      /* scale_idx = 16  (36336 src blocks) */ [ 0xB452,0xAA9F,0xA724,0xA0B4,0x2044,0x26D6,0x2A82,0x3358 ],
      /* scale_idx = 17  (36077 src blocks) */ [ 0xB477,0xAAE4,0xA770,0xA10C,0x1FD0,0x26F6,0x2AB7,0x32C5 ],
      /* scale_idx = 18  (36093 src blocks) */ [ 0xB4D9,0xAB0F,0xA7AC,0xA138,0x2054,0x2740,0x2AEA,0x3368 ],
      /* scale_idx = 19  (36184 src blocks) */ [ 0xB532,0xAB65,0xA7FA,0xA134,0x20A4,0x2784,0x2B21,0x34AB ],
      /* scale_idx = 20  (36038 src blocks) */ [ 0xB4D7,0xABC2,0xA816,0xA160,0x20B4,0x27D0,0x2B7C,0x3495 ],
      /* scale_idx = 21  (36245 src blocks) */ [ 0xB5E6,0xAC04,0xA830,0xA158,0x212C,0x2826,0x2BDD,0x342C ],
      /* scale_idx = 22  (36260 src blocks) */ [ 0xB53A,0xAC2E,0xA86E,0xA1F4,0x2118,0x283B,0x2C13,0x34A4 ],
      /* scale_idx = 23  (35892 src blocks) */ [ 0xB510,0xAC6B,0xA8B6,0xA240,0x2170,0x2879,0x2C38,0x34C9 ],
      /* scale_idx = 24  (36229 src blocks) */ [ 0xB598,0xACA8,0xA8EA,0xA2B4,0x2198,0x28AF,0x2C7F,0x34ED ],
      /* scale_idx = 25  (36251 src blocks) */ [ 0xB593,0xAD19,0xA95A,0xA364,0x2198,0x28D8,0x2CB2,0x3601 ],
      /* scale_idx = 26  (36332 src blocks) */ [ 0xB591,0xAD7D,0xA9F2,0xA42E,0x2210,0x291F,0x2D0D,0x35E3 ],
      /* scale_idx = 27  (36591 src blocks) */ [ 0xB675,0xADD5,0xAA53,0xA494,0x21F8,0x297D,0x2D75,0x360A ],
      /* scale_idx = 28  (36833 src blocks) */ [ 0xB6DC,0xAE72,0xAAF1,0xA4CC,0x2364,0x2A71,0x2E3A,0x3639 ],
      /* scale_idx = 29  (35066 src blocks) */ [ 0xB722,0xB00D,0xAC2E,0xA54A,0x254C,0x2C43,0x3012,0x3746 ],
      /* scale_idx = 30  (47819 src blocks) */ [ 0xB6D2,0xB278,0xB0AF,0xA9D6,0x2ABE,0x3096,0x3263,0x3649 ],
      /* scale_idx = 31  (4897 src blocks) */ [ 0xB55E,0xAE9C,0xABF9,0xA2AC,0x28AA,0x2CBE,0x2F2C,0x33CC ]

    ];

/// CENTROID_TABLE_BITS_K with 16 entries per scale row. Currently built by
/// duplicating the calibrated 8-entry table into adjacent slot pairs
/// (j → 2j, 2j+1); a full 16-entry recalibration is a separate measurement
/// task. The kernel reads `[16]` rows so the type must match.
pub const CENTROID_TABLE_BITS_K: [[u16; 16]; 32] = {
    let mut out = [[0u16; 16]; 32];
    let mut s = 0;
    while s < 32 {
        let mut j = 0;
        while j < 8 {
            out[s][j * 2] = CENTROID_TABLE_BITS_K_OLD[s][j];
            out[s][j * 2 + 1] = CENTROID_TABLE_BITS_K_OLD[s][j];
            j += 1;
        }
        s += 1;
    }
    out
};

#[rustfmt::skip]
    pub const PEAK_CURVE_INDICES_K: [u8; 256] = [
    
          0,  1,  2,  8,  9, 10, 16, 17, 18, 24, 25, 26, 32, 40, 48, 49,
         50, 56, 57, 58, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 98, 99,
        104,105,106,107,112,113,120,121,128,129,130,131,136,137,138,139,
        144,145,146,152,153,154,160,161,168,169,176,177,178,184,185,186,
        192,200,208,216,224,232,240,241,242,243,248,249,250,251, 39, 47,
        215,223,230,238, 38, 46,214,222, 37, 45, 71, 79,119,127,213,221,
         87, 95,195,203,227,235, 23, 31, 36, 44,167,175,212,220,  7, 15,
         35, 43,193,201,211,219,225,233,118,126,183,191, 34, 42,210,218,
         70, 78,151,159, 33, 41, 55, 63,198,206,209,217, 86, 94,166,174,
         22, 30,117,125,196,204,228,236,103,111,247,255, 69, 77,  6, 14,
         85, 93,116,124,165,173,199,207,231,239,182,190, 21, 29, 68, 76,
        150,158,197,205,229,237, 54, 62,135,143,115,123, 84, 92,  3,  4,
          5, 11, 12, 13, 19, 20, 27, 28, 51, 52, 53, 59, 60, 61, 66, 67,
         74, 75, 82, 83, 90, 91,100,101,102,108,109,110,114,122,132,133,
        134,140,141,142,147,148,149,155,156,157,162,163,164,170,171,172,
        179,180,181,187,188,189,194,202,226,234,244,245,246,252,253,254
    
    ];

#[rustfmt::skip]
    pub const PEAK_BIN_OFFSETS_K: [u16; 33] = [
    
          0, 78, 78, 82, 84, 88, 88, 96,102,110,110,120,124,128,132,140,144,144,152,154,156,158,160,162,166,170,172,176,182,186,188,190,256
    
    ];

#[rustfmt::skip]
    pub const SCALE_TABLE_BITS_V: [u16; 32] = [
    
        0x0575,0x168E,0x17CE,0x1850,0x18A3,0x18EA,0x1929,0x1964,
        0x199B,0x19CF,0x1A02,0x1A34,0x1A65,0x1A95,0x1AC6,0x1AF7,
        0x1B29,0x1B5D,0x1B92,0x1BC8,0x1C01,0x1C1F,0x1C3F,0x1C62,
        0x1C88,0x1CB3,0x1CE4,0x1D1E,0x1D67,0x1DCA,0x1E69,0x2008
    
    ];

#[rustfmt::skip]
    pub(super) const CENTROID_TABLE_BITS_V_OLD: [[u16; 8]; 32] = [

      /* scale_idx =  0  (7635 src blocks) */ [ 0xA9F3,0x9FE8,0x9C08,0x9500,0x1340,0x1B10,0x1F40,0x2C53 ],
      /* scale_idx =  1  (43800 src blocks) */ [ 0xB09E,0xA5C2,0xA19C,0x9A40,0x1BF0,0x21F4,0x25E8,0x31D1 ],
      /* scale_idx =  2  (37297 src blocks) */ [ 0xB10F,0xA7B6,0xA3AC,0x9C88,0x1D50,0x2402,0x27C4,0x3220 ],
      /* scale_idx =  3  (36644 src blocks) */ [ 0xB41E,0xA833,0xA454,0x9D50,0x1D78,0x245E,0x2844,0x311D ],
      /* scale_idx =  4  (36307 src blocks) */ [ 0xB26F,0xA88A,0xA4B2,0x9E20,0x1DA0,0x2498,0x2871,0x319C ],
      /* scale_idx =  5  (36217 src blocks) */ [ 0xB33F,0xA8AD,0xA4BA,0x9DC8,0x1E68,0x24F2,0x28B4,0x3214 ],
      /* scale_idx =  6  (36480 src blocks) */ [ 0xB2D5,0xA8F9,0xA534,0x9EA8,0x1E40,0x2516,0x28EE,0x32A6 ],
      /* scale_idx =  7  (35974 src blocks) */ [ 0xB39B,0xA924,0xA55A,0x9EE0,0x1E88,0x2546,0x2934,0x3329 ],
      /* scale_idx =  8  (36106 src blocks) */ [ 0xB487,0xA948,0xA57A,0x9EE8,0x1ED0,0x2560,0x2933,0x3372 ],
      /* scale_idx =  9  (36212 src blocks) */ [ 0xB3FB,0xA974,0xA5B2,0x9F28,0x1EF0,0x2592,0x2967,0x328C ],
      /* scale_idx = 10  (35935 src blocks) */ [ 0xB460,0xA9A7,0xA5CA,0x9EF8,0x1F50,0x25D0,0x29A7,0x3350 ],
      /* scale_idx = 11  (36276 src blocks) */ [ 0xB3CC,0xA9B1,0xA5EA,0x9FA8,0x1F20,0x25DE,0x29B9,0x3482 ],
      /* scale_idx = 12  (35923 src blocks) */ [ 0xB473,0xA9E1,0xA624,0x9FE0,0x1F90,0x2604,0x29CF,0x345B ],
      /* scale_idx = 13  (36137 src blocks) */ [ 0xB359,0xAA09,0xA626,0x9FF0,0x1FB8,0x2628,0x29FC,0x347A ],
      /* scale_idx = 14  (36011 src blocks) */ [ 0xB4AA,0xAA25,0xA666,0xA034,0x1FC8,0x263A,0x2A16,0x345F ],
      /* scale_idx = 15  (36307 src blocks) */ [ 0xB4AB,0xAA47,0xA6A0,0xA054,0x1F68,0x2640,0x2A18,0x33E8 ],
      /* scale_idx = 16  (35915 src blocks) */ [ 0xB437,0xAA53,0xA69E,0xA064,0x2000,0x2676,0x2A38,0x34BF ],
      /* scale_idx = 17  (36203 src blocks) */ [ 0xB4DB,0xAA76,0xA6C8,0xA080,0x1F80,0x268C,0x2A7A,0x34E7 ],
      /* scale_idx = 18  (35969 src blocks) */ [ 0xB478,0xAAA2,0xA6D2,0xA054,0x2038,0x26D0,0x2A95,0x3564 ],
      /* scale_idx = 19  (36317 src blocks) */ [ 0xB4E4,0xAAC1,0xA704,0xA094,0x201C,0x26C0,0x2AA6,0x34E3 ],
      /* scale_idx = 20  (35929 src blocks) */ [ 0xB4AA,0xAAD5,0xA700,0xA074,0x2034,0x26EA,0x2AD4,0x35EA ],
      /* scale_idx = 21  (36227 src blocks) */ [ 0xB4C6,0xAAF8,0xA760,0xA0E0,0x2044,0x26E8,0x2ACB,0x3516 ],
      /* scale_idx = 22  (36215 src blocks) */ [ 0xB48E,0xAB21,0xA774,0xA0D8,0x2068,0x2732,0x2B00,0x34F1 ],
      /* scale_idx = 23  (36199 src blocks) */ [ 0xB51A,0xAB47,0xA786,0xA0BC,0x2084,0x2754,0x2B3F,0x34A4 ],
      /* scale_idx = 24  (36183 src blocks) */ [ 0xB4D3,0xAB64,0xA7A8,0xA108,0x2084,0x275C,0x2B4D,0x3484 ],
      /* scale_idx = 25  (36006 src blocks) */ [ 0xB4F6,0xAB82,0xA7A2,0xA0C4,0x20D0,0x27BC,0x2B7A,0x350C ],
      /* scale_idx = 26  (36250 src blocks) */ [ 0xB4BC,0xABB0,0xA808,0xA138,0x2094,0x27BA,0x2B93,0x3555 ],
      /* scale_idx = 27  (36361 src blocks) */ [ 0xB51A,0xAC10,0xA834,0xA15C,0x20E8,0x2812,0x2BEA,0x35C0 ],
      /* scale_idx = 28  (36642 src blocks) */ [ 0xB560,0xAC3C,0xA851,0xA1B4,0x2120,0x282C,0x2C27,0x3548 ],
      /* scale_idx = 29  (37210 src blocks) */ [ 0xB4D0,0xAC78,0xA886,0xA188,0x21C8,0x2883,0x2C6F,0x34C8 ],
      /* scale_idx = 30  (38828 src blocks) */ [ 0xB4DC,0xAD2F,0xA921,0xA27C,0x21D4,0x2902,0x2D19,0x3436 ],
      /* scale_idx = 31  (12493 src blocks) */ [ 0xB457,0xABCD,0xA844,0xA0F8,0x21E0,0x285B,0x2BD9,0x3435 ]

    ];

/// CENTROID_TABLE_BITS_V — 16-entry table built by duplicating the 8-entry
/// calibration. See the note on the K-side equivalent.
pub const CENTROID_TABLE_BITS_V: [[u16; 16]; 32] = {
    let mut out = [[0u16; 16]; 32];
    let mut s = 0;
    while s < 32 {
        let mut j = 0;
        while j < 8 {
            out[s][j * 2] = CENTROID_TABLE_BITS_V_OLD[s][j];
            out[s][j * 2 + 1] = CENTROID_TABLE_BITS_V_OLD[s][j];
            j += 1;
        }
        s += 1;
    }
    out
};

#[rustfmt::skip]
    pub const PEAK_CURVE_INDICES_V: [u8; 256] = [
    
          0,  1,  2,  8,  9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34, 35,
         40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59, 64, 65, 66, 72,
         73, 74, 80, 81, 82, 88, 89, 90, 96, 97,104,105,112,113,114,115,
        120,121,122,123,128,129,130,136,137,138,144,145,152,153,160,161,
        162,166,167,168,169,170,174,175,176,177,184,185,192,200,208,209,
        216,217,224,225,226,227,228,229,230,231,232,233,234,235,236,237,
        238,239,240,248,163,164,165,171,172,173,199,207,246,254,198,206,
        151,159,197,205,215,223,103,111,243,251, 71, 79,183,191,196,204,
          7, 15,195,203,241,249,135,143,214,222,194,202, 23, 31,150,158,
         87, 95,193,201,102,110,182,190, 70, 78,213,221,244,252, 55, 63,
        119,127,149,157,  6, 14,101,109,181,189,212,220,247,255,134,142,
         69, 77,148,156, 22, 30,245,253, 39, 47, 86, 94,211,219,100,108,
          3,  4,  5, 11, 12, 13, 19, 20, 21, 27, 28, 29, 36, 37, 38, 44,
         45, 46, 52, 53, 54, 60, 61, 62, 67, 68, 75, 76, 83, 84, 85, 91,
         92, 93, 98, 99,106,107,116,117,118,124,125,126,131,132,133,139,
        140,141,146,147,154,155,178,179,180,186,187,188,210,218,242,250
    
    ];

#[rustfmt::skip]
    pub const PEAK_BIN_OFFSETS_V: [u16; 33] = [
    
          0,100,106,108,110,112,112,118,122,128,128,134,138,140,144,148,152,152,158,160,162,164,166,168,172,174,176,180,184,188,190,192,256
    
    ];
