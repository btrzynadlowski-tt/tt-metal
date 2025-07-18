#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    uint32_t is_egress = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_addr = get_arg_val<uint32_t>(2);

    // Size of a single tile of floats
    const uint32_t single_tile_size = sizeof(float) * 32 * 32;

    // Get DRAM address
    const InterleavedAddrGenFast<true> dram_buffer = {
        .bank_base_address = dram_addr,      // The base address of the buffer
        .page_size = single_tile_size,       // The size of a buffer page
        .data_format = DataFormat::Float32,  // The data format of the buffer
    };

    // Circular buffer for receiving the two operands and the result buffer (3 tiles)
    constexpr uint32_t cb_id_in = tt::CBIndex::c_1;
    
    // Copy data between SRAM <-> DRAM depending on the mode
    if (!is_egress) {
        // Egress mode: DRAM -> SRAM (*host* sending *to* kernels)
        cb_reserve_back(cb_id_in, 3);  // get 2 input operands and the space for the result
        uint32_t l1_buffer_addr = get_write_ptr(cb_id_in);
        uint32_t operand1_addr = l1_buffer_addr;
        uint32_t operand2_addr = operand1_addr + single_tile_size;
        noc_async_read_tile(0, dram_buffer, operand1_addr);
        noc_async_read_tile(1, dram_buffer, operand2_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in, 3);

        // Debug
        float* src = reinterpret_cast<float*>(l1_buffer_addr);
        DPRINT << "float=" << src[0] << ENDL();
    } else {
        // Ingress mode: SRAM -> DRAM (*host* retrieving *from* kernels)
        cb_wait_front(cb_id_in, 3);
        uint32_t l1_buffer_addr = get_read_ptr(cb_id_in);

        // For now, copy operand 1 to the result (and add 0.5f to everything)
        uint32_t result_addr = l1_buffer_addr + 2 * single_tile_size;
        float* src = reinterpret_cast<float*>(l1_buffer_addr);
        float* dest = reinterpret_cast<float*>(result_addr);
        for (uint32_t i = 0; i < single_tile_size / sizeof(float); i++) {
            *dest++ = *src++ + 0.5f;
        }

        // Write result
        noc_async_write_tile(2, dram_buffer, result_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_id_in, 3);
    }
}
