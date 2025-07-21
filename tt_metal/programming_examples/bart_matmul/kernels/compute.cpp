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

namespace NAMESPACE {
static inline void kernel() {
    constexpr auto cb_to_compute_id = tt::CBIndex::c_1;     // from ingress processor to here
    constexpr auto cb_from_compute_id = tt::CBIndex::c_16;  // from here to egress processor

    const uint32_t single_tile_elements = 32 * 32;
    const uint32_t single_tile_size = sizeof(float) * single_tile_elements;

    PRINT(DPRINT << "got here 1" << ENDL());
    
    // Reserve space for result
    //cb_reserve_back(cb_from_compute_id, 1);
    llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(cb_from_compute_id);
    llk_pack_init(cb_from_compute_id);
    llk_pack_dest_init<DST_ACCUM_MODE, false>();
    llk_wait_for_free_tiles<false, false, false>(cb_from_compute_id, 1);
    uint32_t result_addr = /*108096;*/get_local_cb_interface(cb_from_compute_id).fifo_wr_ptr - 0;

    PRINT(DPRINT << "got here 2" << ENDL());

    // Wait for ingress processor to hand us two operands
    //cb_wait_front(cb_to_compute_id, 2);
    llk_wait_tiles(cb_to_compute_id, 2);
    uint32_t l1_operand_buffer_addr = /*99904;*/get_local_cb_interface(cb_to_compute_id).fifo_rd_ptr - 0;
    uint32_t operand1_addr = l1_operand_buffer_addr;
    uint32_t operand2_addr = operand1_addr + single_tile_size;

    PRINT(DPRINT << "Compute: read operand from=" << operand1_addr << ", write result to=" << result_addr << ENDL());
    
    // Copy operand 1 to result for now and add 0.25f everywhere
    //volatile tt_l1_ptr float *src = reinterpret_cast<volatile float *>(operand1_ptr);
    //volatile tt_l1_ptr float *dest = reinterpret_cast<volatile float *>(result_ptr);
    float *src = reinterpret_cast<float *>(operand1_addr);
    float *dest = reinterpret_cast<float *>(result_addr);
    for (uint32_t i = 0; i < 4 * 4; i++) {
        *dest++ = *src++ + 0.25f;
    }

    // Push result to egress data movement processor
    cb_push_back(cb_from_compute_id, 1);

    // Finished with input
    cb_pop_front(cb_to_compute_id, 2);
}

void MAIN {
    constexpr auto cb_to_compute = tt::CBIndex::c_1;    // from ingress processor to here
    constexpr auto cb_from_compute = tt::CBIndex::c_2;  // from here to egress processor
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    PROCESSOR(kernel());
}
}  // namespace NAMESPACE
