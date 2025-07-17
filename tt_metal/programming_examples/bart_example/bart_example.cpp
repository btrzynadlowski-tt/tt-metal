#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = sizeof(float) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    // Create a buffer in SRAM for kernel working memory
    tt::tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1};

    std::shared_ptr<Buffer> l1_buffer = CreateBuffer(l1_config);

    /*
     * Create 2 4x4 matrices:
     *
     *      1   2  3  4   17 18 19 20
     *      5   6  7  8 * 21 22 23 24
     *      9  10 11 12   25 26 27 28
     *      13 14 15 16   29 30 31 32
     *
     * They will be layed out linearly, one after the other, in row-major form
     * (i.e., like a video buffer).
     *
     * The result will be copied into the space after these as a third matrix.
     * Expected result:
     *
     *      250   260  270  280
     *      618   644  670  696
     *      986  1028 1070 1112
     *      1354 1412 1470 1528
     */

    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> dram_buffer = CreateBuffer(dram_config);
    size_t operand_1_idx = 0 * 4 * 4;
    size_t operand_2_idx = 1 * 4 * 4;
    size_t result_idx = 2 * 4 * 4;

    std::vector<float> input_data(4 * 4 * 2, 0);
    for (size_t i = 0; i < 4 * 4; i++) {
        input_data[operand_1_idx + i] = float(i + 1);
    }
    for (size_t i = 0; i < 4 * 4; i++) {
        input_data[operand_2_idx + i] = float(i + 1 + 16);
    }

    // Copy to device DRAM
    EnqueueWriteBuffer(cq, dram_buffer, input_data, false);

    // Create a kernel that will perform the operation
    KernelHandle kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bart_example/kernels/bart_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set program arguments: locations of the buffers
    const std::vector<uint32_t> runtime_args = {l1_buffer->address(), dram_buffer->address()};
    SetRuntimeArgs(program, kernel_id, core, runtime_args);

    // Run the program and wait for it to finish before reading memory back
    EnqueueProgram(cq, program, /*blocking=*/false);
    Finish(cq);
    std::vector<float> result;
    EnqueueReadBuffer(cq, dram_buffer, result, /*blocking*/ true);

    // Print and validate
    std::vector<float> expected{250, 260, 270, 280, 618, 644, 670, 696, 986, 1028, 1070, 1112, 1354, 1412, 1470, 1528};
    size_t i = 0;
    bool pass = true;
    for (size_t y = 0; y < 4; y++) {
        for (size_t x = 0; x < 4; x++) {
            std::cout << result[result_idx + y * 4 + x] << ' ';
            pass &= result[result_idx + y * 4 + x] == expected[i];
            i++;
        }
        std::cout << std::endl;
    }

    std::cout << "Test " << (pass ? "passed" : "FAILED") << std::endl;
    return 0;
}
