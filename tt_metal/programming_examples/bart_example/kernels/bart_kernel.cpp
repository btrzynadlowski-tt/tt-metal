void kernel_main() {
    uint32_t l1_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_addr = get_arg_val<uint32_t>(1);

    // Get DRAM address
    const uint32_t tile_size_bytes = sizeof(float) * 32 * 32;
    const InterleavedAddrGenFast<true> dram_buffer = {
        .bank_base_address = dram_addr,        // The base address of the buffer
        .page_size = tile_size_bytes,          // The size of a buffer page
        .data_format = DataFormat::Float16_b,  // The data format of the buffer
    };

    // Pointer to SRAM
    float* l1_buffer = reinterpret_cast<float*>(l1_addr);

    // Read tile from DRAM into SRAM. NOC (only accessible to dataflow processors)
    // is used because processors do not see DRAM, which is external to the device.
    // Processors see SRAM directly but NOC is used to transport data from DRAM into
    // SRAM, where all processors in a core can see it.
    noc_async_read_tile(0, dram_buffer, l1_addr);
    noc_async_read_barrier();
    noc_async_write_barrier();

    // Perform matrix multiplication the old fashioned way: one scalar op at a time
    uint32_t operand_0_idx = 0;
    uint32_t operand_1_idx = 1 * 4 * 4;
    uint32_t result_idx = 2 * 4 * 4;
    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col_b = 0; col_b < 4; col_b++) {
            l1_buffer[result_idx + row * 4 + col_b] = 0.0f;
            for (uint32_t col = 0; col < 4; col++) {
                l1_buffer[result_idx + row * 4 + col_b] +=
                    l1_buffer[operand_0_idx + row * 4 + col] * l1_buffer[operand_1_idx + col * 4 + col_b];
            }
        }
    }

    // Write back to DRAM
    noc_async_write_tile(0, dram_buffer, l1_addr);
    noc_async_write_barrier();
}
