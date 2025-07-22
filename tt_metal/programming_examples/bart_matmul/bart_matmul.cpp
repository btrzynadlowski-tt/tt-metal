/*
 * bart_matmul.cpp
 * Matrix multiplication using RISC-V instruction set on compute processor.
 * 
 * This example builds off of bart_example to perform a matrix multiplication on a compute
 * processor. As before, we perform the multiplication using the RISC-V instruction set but on a
 * compute processor this time rather than a data movement processor.
 * 
 * We use the processors in this configuration:
 * 
 *      1. Data Movement Processor (Ingress): Use NOC to copy two operands (each a 4x4 matrix) to
 *         SRAM. Pass SRAM buffer address to the unpack compute processor via a circular buffer
 *         (cb_to_compute).
 *      2. Compute Processor (Unpack): Receives the SRAM address of the two 4x4 matrices from the
 *         ingress processor. Invokes a function (that involves semaphores and can be invoked on
 *         all three compute threads -- unpack, math, pack) that obtains the raw address.
 *      3. Compute Processor (Math): Does nothing other than invoke the function that obtains the
 *         buffer address.
 *      4. Compute Processor (Pack): Obtains the operand address and then performs the actual
 *         matmul. Uses a second circular buffer (cb_from_compute) to store the output as a tile in
 *         SRAM.
 *      5. Data Movement Processor (Egress): Uses the cb_from_compute circular buffer to receive
 *         the 4x4 result matrix from pack and ships it to DRAM via the NOC, for the host to
 *         access.
 *
 * The 3 compute processors are intended to be programmed using a single kernel. Internally, many
 * of the functions use UNPACK(), MATH(), and PACK() macros to selectively enable code depending on
 * which specific processor they are running on. The compute APIs are designed to use the FPU/SFPU
 * and load data tiles into special registers. The circular buffer APIs differ between compute and
 * data movement processors and are intended to be used in a very specific way to load data into
 * registers. Therefore, a given circular buffer function will not do the same thing on pack, math,
 * and unpack.
 * 
 * This demo abuses the APIs to move data manually. Using these APIs, it does not seem possible to
 * use a single compute processor (e.g., math or pack), hence why both unpack and pack are
 * involved. Theoretically, manual management of SRAM and custom circular buffer implementations
 * are perfectly possible but we try to use existing functions here.
 * 
 * We store the 4x4 matrices in linear form one row at a time. Each number is a float, so each
 * matrix takes 16 dwords but uses a whole tile for storage. A tile is 32x32 (1024) dwords. We
 * ignore everything after the initial 16 dwords. 
 * 
 * We structure the two operands this way on the host and transfer them to a DRAM buffer on the TT
 * device. After our kernels run, a single tile of output is copied back.
 */


#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static std::vector<float> matrix_multiply_4x4(const std::vector<float> &input_data, size_t operand_1_idx, size_t operand_2_idx) {
    std::vector<float> result(4*4);
    for (uint32_t row = 0; row < 4; row++) {
        for (uint32_t col_b = 0; col_b < 4; col_b++) {
            result[row * 4 + col_b] = 0.0f;
            for (uint32_t col = 0; col < 4; col++) {
                float a = input_data[operand_1_idx + row * 4 + col];
                float b = input_data[operand_2_idx + col * 4 + col_b];
                result[row * 4 + col_b] += a * b;
            }
        }
    }
    return result;
}

int main() {
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_elements = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t single_tile_size = sizeof(float) * single_tile_elements;
    constexpr uint32_t operand_buffer_size = 2 * single_tile_size;  // operands A and B stored in separate tiles
    constexpr uint32_t result_buffer_size = 1 * single_tile_size;   // result
    constexpr uint32_t buffer_size = operand_buffer_size + result_buffer_size;  // buffer holds both operands and result contiguously

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
        .device = device, .size = buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> dram_buffer = CreateBuffer(dram_config);
    size_t operand_1_idx = 0 * single_tile_elements;
    size_t operand_2_idx = 1 * single_tile_elements;
    size_t result_idx = 2 * single_tile_elements;

    std::vector<float> input_data(buffer_size, 0);
    for (size_t i = 0; i < 4 * 4; i++) {
        input_data[operand_1_idx + i] = float(i + 1);
    }
    for (size_t i = 0; i < 4 * 4; i++) {
        input_data[operand_2_idx + i] = float(i + 1 + 16);
    }

    // Copy to device DRAM
    EnqueueWriteBuffer(cq, dram_buffer, input_data, false);

    /*
     * Create circular buffers so that we can push data from one processor (e.g., the ingress data
     * movement processor -> compute processor -> egress data movement processor) to the other.
     */
    constexpr uint32_t cb_to_compute_index = CBIndex::c_1;
    CircularBufferConfig cb_to_compute_config = CircularBufferConfig(
        operand_buffer_size,    // this CB only transfers the operands from ingress -> compute
        {{cb_to_compute_index, tt::DataFormat::Float32}}
    ).set_page_size(cb_to_compute_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_to_compute_config);

    constexpr uint32_t cb_from_compute_index = CBIndex::c_16;
    CircularBufferConfig cb_from_compute_config = CircularBufferConfig(
        result_buffer_size,     // this CB is used to pass the result from compute -> egress
        {{cb_from_compute_index, tt::DataFormat::Float32}}
    ).set_page_size(cb_from_compute_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_from_compute_config);

    /*
     * Create data movement kernels that will copy from DRAM to SRAM and also support SRAM to
     * DRAM (from the same kernel code, controlled by program args, just to be cute).
     */

    KernelHandle data_ingress_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bart_matmul/kernels/data.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle data_egress_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bart_matmul/kernels/data.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    const std::vector<uint32_t> data_ingress_runtime_args = {
        0,  // ingress mode: copy from DRAM -> SRAM
        dram_buffer->address()};

    const std::vector<uint32_t> data_egress_runtime_args = {
        1,  // egress mode: copy from SRAM -> DRAM
        dram_buffer->address()};

    SetRuntimeArgs(program, data_ingress_kernel_id, core, data_ingress_runtime_args);
    SetRuntimeArgs(program, data_egress_kernel_id, core, data_egress_runtime_args);

    /*
     * Create compute kernel that will perform the matrix multiplication
     */

    std::vector<uint32_t> compute_kernel_args = {};

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bart_matmul/kernels/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );

    // Run the program and wait for it to finish before reading memory back
    EnqueueProgram(cq, program, /*blocking=*/false);
    Finish(cq);
    std::vector<float> result;
    EnqueueReadBuffer(cq, dram_buffer, result, /*blocking*/ true);

    // Print and validate
    std::vector<float> expected = matrix_multiply_4x4(input_data, operand_1_idx, operand_2_idx);//{250, 260, 270, 280, 618, 644, 670, 696, 986, 1028, 1070, 1112, 1354, 1412, 1470, 1528};
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
