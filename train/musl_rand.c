/* musl's rand_r, verbatim semantics — the emscripten wasm build links
 * musl libc, so the native training build must use the SAME rand_r for
 * bit-exact parity with the league sim (macOS libc rand_r differs).
 * Compiled with -Drand_r=cogame_musl_rand_r on every sim TU. */

static unsigned temper(unsigned x) {
    x ^= x >> 11;
    x ^= x << 7 & 0x9D2C5680;
    x ^= x << 15 & 0xEFC60000;
    x ^= x >> 18;
    return x;
}

int cogame_musl_rand_r(unsigned *seed) {
    return temper(*seed = *seed * 1103515245 + 12345) / 2;
}
