/* CPU event simulator. No fast-math or fused multiply-add: retain Python's
 * event thresholds, rates, simultaneous retirements and ascending admission.
 * This file has no Python, NumPy, GPU or third-party build dependencies. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define EPS_KEYS 1e-6
#define EPS_US 1e-9

/* NumPy's contiguous float64 reduction uses eight accumulators in blocks of
 * 128, then a balanced recursive split. Retaining this addition order avoids
 * perturbing bandwidth-cap transitions solely by changing the sum algorithm. */
static double pairwise_sum(const double *a, size_t n) {
    if (n < 8) {
        double result = -0.0;
        for (size_t i = 0; i < n; ++i) result += a[i];
        return result;
    }
    if (n <= 128) {
        double r[8];
        for (size_t j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (size_t j = 0; j < 8; ++j) r[j] += a[i + j];
        double result = ((r[0] + r[1]) + (r[2] + r[3]))
                      + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) result += a[i];
        return result;
    }
    size_t middle = n / 2;
    middle -= middle % 8;
    return pairwise_sum(a, middle) + pairwise_sum(a + middle, n - middle);
}

/* Return 0: success; 1: event guard; 2: nonfinite arithmetic; 3: allocation. */
int kernelscope_simulate(const double *keys, size_t n, size_t n_sm,
                        size_t slots_per_sm, const int64_t *sm_order,
                        double cost, double startup, double empty, double gamma,
                        double bytes_per_key, double bandwidth, size_t max_events,
                        double *out) {
    size_t slots = n_sm * slots_per_sm;
    double *fixed = calloc(slots, sizeof(double));
    double *remaining = calloc(slots, sizeof(double));
    double *rates = calloc(slots, sizeof(double));
    size_t *slot_sm = calloc(slots, sizeof(size_t));
    size_t *residents = calloc(n_sm, sizeof(size_t));
    uint8_t *occupied = calloc(slots, sizeof(uint8_t));
    uint8_t *phase = calloc(slots, sizeof(uint8_t));
    int status = 0;
    if (!fixed || !remaining || !rates || !slot_sm || !residents || !occupied || !phase) {
        status = 3;
        goto cleanup;
    }
    size_t next = 0, active = 0, events = 0;
    double elapsed = 0.0, bw_elapsed = 0.0;
    for (size_t i = 0; i < slots; ++i) {
        slot_sm[i] = (size_t)sm_order[i % n_sm];
        if (next < n) {
            remaining[i] = keys[next++];
            fixed[i] = remaining[i] > 0.0 ? startup : empty;
            occupied[i] = 1;
            residents[slot_sm[i]]++;
            active++;
        }
    }
    while (active) {
        if (++events > max_events) { status = 1; goto cleanup; }
        double dt_fixed = INFINITY, dt_keys = INFINITY;
        int done_now = 0;
        for (size_t i = 0; i < slots; ++i) {
            rates[i] = 0.0;
            phase[i] = 0;
            if (!occupied[i]) continue;
            if (fixed[i] > EPS_US) {
                phase[i] = 1;
                if (fixed[i] < dt_fixed) dt_fixed = fixed[i];
            } else if (remaining[i] > EPS_KEYS) {
                phase[i] = 2;
                rates[i] = (residents[slot_sm[i]] >= 2 ? gamma : 1.0) / cost;
            } else {
                done_now = 1;
            }
        }
        double demand = pairwise_sum(rates, slots) * bytes_per_key;
        if (!isfinite(demand)) { status = 2; goto cleanup; }
        double scale = demand <= bandwidth ? 1.0 : bandwidth / demand;
        for (size_t i = 0; i < slots; ++i) {
            rates[i] *= scale;
            if (phase[i] == 2) {
                double duration = remaining[i] / rates[i];
                if (duration < dt_keys) dt_keys = duration;
            }
        }
        double dt = done_now ? 0.0 : fmin(dt_fixed, dt_keys);
        elapsed += dt;
        if (scale < 1.0) bw_elapsed += dt;
        if (!isfinite(elapsed) || !isfinite(bw_elapsed) || dt < 0.0) {
            status = 2;
            goto cleanup;
        }
        for (size_t i = 0; i < slots; ++i) {
            if (phase[i] == 1) fixed[i] -= dt;
            if (phase[i] == 2) remaining[i] -= rates[i] * dt;
            if (occupied[i] && fixed[i] <= EPS_US && remaining[i] <= EPS_KEYS) {
                occupied[i] = 0;
                residents[slot_sm[i]]--;
                active--;
                if (next < n) {
                    remaining[i] = keys[next++];
                    fixed[i] = remaining[i] > 0.0 ? startup : empty;
                    occupied[i] = 1;
                    residents[slot_sm[i]]++;
                    active++;
                }
            }
        }
    }
    out[0] = elapsed;
    out[1] = bw_elapsed;
    out[2] = (double)events;
cleanup:
    free(fixed); free(remaining); free(rates); free(slot_sm);
    free(residents); free(occupied); free(phase);
    return status;
}
