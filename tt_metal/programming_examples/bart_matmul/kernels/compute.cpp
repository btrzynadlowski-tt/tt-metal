//TODO: need new description that mentions how we abuse the APIs to use 2 CBs rather than three,
//.     and how we perform math with the RISC-V ISA on the unpack processor rather than FPU/SFPU in
//.     a math kernel.
/*
 * Compute kernel. This runs on all 3 compute processors. This could normally introduce data races,
 * as each kernel is accessing the same regions of memory but this code is safe because the matrix
 * multiplications simply read the operands and then write the result without reading intermediate
 * results anywhere. So all 3 kernels will produce the same output and contention doesn't matter.
 */

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

#include "llk_io_pack.h"
#include "llk_io_unpack.h"
#include "llk_pack_api.h"

#define CONCAT(a, b) a##b
#define EXPAND_CONCAT(a, b) CONCAT(a, b)

#define PROCESSOR UNPACK
#define PRINT EXPAND_CONCAT(DPRINT_, PROCESSOR)

constexpr uint32_t single_tile_elements = 32 * 32;
constexpr uint32_t single_tile_size = sizeof(float) * single_tile_elements;

constexpr auto cb_to_compute_id = tt::CBIndex::c_1;     // from ingress processor to here
constexpr auto cb_from_compute_id = tt::CBIndex::c_16;  // from here to egress processor

namespace NAMESPACE {

static inline void matrix_multiply_4x4(volatile float *result, volatile float *a, volatile float *b) {
    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col_b = 0; col_b < 4; col_b++) {
            result[row * 4 + col_b] = 0.0f;
            for (uint32_t col = 0; col < 4; col++) {
                result[row * 4 + col_b] += a[row * 4 + col] * b[col * 4 + col_b];
            }
        }
    }
}

void kernel_unpack() {
    // Wait for ingress processor to hand us two operands.
    // NOTE: cb_get_tile pointer is off by -16 bytes, and we manually need to advance. See
    // embedding_backward.cpp where this comment exists: "Need to shift because read ptr is off by 1 << 4 in BBE"
    // cb_get_tile()/cb_release_tile() must be called on all three compute processors. It uses
    // hardware semaphores internally. Skipping one get/release pair will cause another processor
    // to hang!
    cb_wait_front(cb_to_compute_id, 2);
    volatile float *l1_operand_ptr = nullptr;
    cb_get_tile(cb_to_compute_id, 0, &l1_operand_ptr);
    l1_operand_ptr += 4;
    DPRINT_UNPACK(DPRINT << "unpack: l1_operand_ptr=" << reinterpret_cast<uint32_t>(l1_operand_ptr) << ENDL());
    DPRINT_UNPACK(DPRINT << "unpack: l1_operand_ptr[0]=" << l1_operand_ptr[0] << ", l1_operand_ptr[1]=" << l1_operand_ptr[1] << ENDL());
    cb_release_tile(cb_to_compute_id);
    DPRINT_UNPACK(DPRINT << "unpack: released tile" << ENDL());

    // Finished with input
    cb_pop_front(cb_to_compute_id, 2);
    DPRINT_UNPACK(DPRINT << "unpack: finished" << ENDL());
}

void kernel_math() {
    // The cb_get_tile function executes on all compute processors and passes the pointer to all. 
    // We must use it symmetrically on all 3 compute processors to avoid hanging the others.
    // This math kernel doesn't do any actual work.
    volatile float *l1_operand_ptr = nullptr;
    cb_get_tile(cb_to_compute_id, 0, &l1_operand_ptr);
    l1_operand_ptr += 4;
    DPRINT_MATH(DPRINT << "math: l1_operand_ptr=" << reinterpret_cast<uint32_t>(l1_operand_ptr) << ENDL());
    DPRINT_MATH(DPRINT << "math: l1_operand_ptr[0]=" << l1_operand_ptr[0] << ", l1_operand_ptr[1]=" << l1_operand_ptr[1] << ENDL());
    cb_release_tile(cb_to_compute_id);
    DPRINT_MATH(DPRINT << "math: finished" << ENDL());
}

void kernel_pack() {
    // We use the pack processor to perform the matrix multiplication on the data received by the
    // unpack processor. And then we ship it off to the egress data movement processor using the
    // second queue. 

    // The cb_get_tile function executes on all compute processors and passes the pointer to all. So we can receive it here.
    volatile float *l1_operand_ptr = nullptr;
    cb_get_tile(cb_to_compute_id, 0, &l1_operand_ptr);
    l1_operand_ptr += 4;
    DPRINT_PACK(DPRINT << "pack: l1_operand_ptr=" << reinterpret_cast<uint32_t>(l1_operand_ptr) << ENDL());
    DPRINT_PACK(DPRINT << "pack: l1_operand_ptr[0]=" << l1_operand_ptr[0] << ", l1_operand_ptr[1]=" << l1_operand_ptr[1] << ENDL());
    cb_release_tile(cb_to_compute_id);

    // Get space for output
    cb_reserve_back(cb_from_compute_id, 1);
    DPRINT_PACK(DPRINT << "pack: reserved space for result" << ENDL());

    // Get pointer to write to
    volatile float *l1_result_ptr = nullptr;
    l1_result_ptr = reinterpret_cast<float *>(get_local_cb_interface(cb_from_compute_id).fifo_wr_ptr * 16); // not quite sure why * 16 is needed...
    DPRINT_PACK(DPRINT << "pack: l1_result_ptr=" << reinterpret_cast<uint32_t>(l1_result_ptr) << ENDL());

    // Compute matrix result
    volatile float *operand_a = &l1_operand_ptr[0];
    volatile float *operand_b = &l1_operand_ptr[single_tile_elements];
    volatile float *dest = l1_result_ptr;
    matrix_multiply_4x4(dest, operand_a, operand_b);

    // Push result to egress data movement processor
    cb_push_back(cb_from_compute_id, 1);

    DPRINT_PACK(DPRINT << "pack: finished" << ENDL());
}

void MAIN {
    binary_op_init_common(cb_to_compute_id, tt::CBIndex::c_2, cb_from_compute_id);

    UNPACK(kernel_unpack());
    MATH(kernel_math());
    PACK(kernel_pack());
}
}  // namespace NAMESPACE
