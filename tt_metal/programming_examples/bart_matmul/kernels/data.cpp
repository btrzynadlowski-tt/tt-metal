/*
 * Data movement kernel. Used on both data movement processors, for ingress (DRAM -> SRAM) and 
 * egress (SRAM -> DRAM). The mode is controlled by a runtime arg.
 */

#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    uint32_t is_egress = get_arg_val<uint32_t>(0);
    uint32_t dram_addr = get_arg_val<uint32_t>(1);

    // Size of a single tile of floats
    const uint32_t single_tile_size = sizeof(float) * 32 * 32;

    // Get DRAM address
    const InterleavedAddrGenFast<true> dram_buffer = {
        .bank_base_address = dram_addr,      // The base address of the buffer
        .page_size = single_tile_size,       // The size of a buffer page
        .data_format = DataFormat::Float32,  // The data format of the buffer
    };

    // Circular buffer to send operands received via NOC to compute kernel
    constexpr uint32_t cb_to_compute_id = tt::CBIndex::c_1;

    // Circular buffer to send result computed in compute kernel to egress data processor (for
    // subsequent output to DRAM via NOC)
    constexpr uint32_t cb_from_compute_id = tt::CBIndex::c_16;
    
    // Copy data between SRAM <-> DRAM depending on the mode
    if (!is_egress) {
        // Ingress mode: DRAM -> SRAM (host sends to DRAM, ingress data movement processor pushes
        // to compute kernel via CB)
        DPRINT << "got here 1" << ENDL();
        cb_reserve_back(cb_to_compute_id, 2);  // get 2 input operands
        uint32_t l1_buffer_addr = get_write_ptr(cb_to_compute_id);
        uint32_t operand1_addr = l1_buffer_addr;
        uint32_t operand2_addr = operand1_addr + single_tile_size;
        noc_async_read_tile(0, dram_buffer, operand1_addr);
        noc_async_read_tile(1, dram_buffer, operand2_addr);
        noc_async_read_barrier();
        cb_push_back(cb_to_compute_id, 2);

        // Debug
        float* src = reinterpret_cast<float*>(l1_buffer_addr);
        DPRINT << "float=" << src[0] << ENDL();
    } else {
        // Egress mode: SRAM -> DRAM (egress data movement process pulls from compute kernel via a
        // second CB and then writes to DRAM)
        cb_wait_front(cb_from_compute_id, 1);
        uint32_t l1_buffer_addr = get_read_ptr(cb_from_compute_id);

        // Write result to third tile (2) of DRAM buffer
        noc_async_write_tile(2, dram_buffer, l1_buffer_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_from_compute_id, 1);
    }
}
