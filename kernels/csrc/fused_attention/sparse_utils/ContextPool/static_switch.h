#define POOL_SWITCH(POOL_SIZE, ...)        \
  [&] {                                    \
    if (POOL_SIZE == 16) {                 \
      constexpr static int PoolBlock = 16; \
      return __VA_ARGS__();                \
    } else if (POOL_SIZE == 32) {          \
      constexpr static int PoolBlock = 32; \
      return __VA_ARGS__();                \
    }                                      \
  }()


#define HEADDIM_SWITCH(HEAD_SIZE, ...)       \
  [&] {                                      \
    if (HEAD_SIZE == 128) {                  \
      constexpr static int HeadDim = 128;    \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define KVZERO_SWITCH(KV_WITH_ZEROS, ...)      \
  [&] {                                        \
    if (KV_WITH_ZEROS) {                       \
      constexpr static bool EnableZero = true; \
      return __VA_ARGS__();                    \
    }                                          \
  }()


