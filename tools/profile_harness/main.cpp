#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "acl/acl.h"

#ifdef PROFILE_PIVOT
extern void sqrt_softplus_pivot_top_k_custom_do(
    uint32_t blockDim,
    void* stream,
    uint8_t* routerLogits,
    uint8_t* topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum);
#else
extern void sqrt_softplus_prefix_merge_top_k_dispatch_custom_do(
    uint32_t blockDim,
    void* stream,
    uint8_t* routerLogits,
    uint8_t* topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum);
#endif

namespace {

bool CheckAcl(const char* action, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ACL ERROR] %s: %d\n", action, static_cast<int>(ret));
    return false;
}

float ReferenceTransform(float x)
{
    const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
    return std::sqrt(softplus);
}

void FillInput(float* input, int32_t rows, int32_t experts, int32_t k)
{
    for (int32_t row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(experts);
        for (int32_t i = 0; i < experts; ++i) {
            const int32_t raw =
                (i * 37 + experts * 13 + k * 7 + row * 53) % 257 - 128;
            input[base + static_cast<size_t>(i)] =
                static_cast<float>(raw) / 8.0f +
                static_cast<float>(i) * 1.0e-6f +
                static_cast<float>(row) * 1.0e-4f;
        }
        input[base] = 25.0f + static_cast<float>(row) * 0.25f;
        if (experts > 1) {
            input[base + 1] = -INFINITY;
        }
    }
}

bool RunOne(aclrtStream stream, int32_t rows, int32_t experts, int32_t k)
{
    const size_t inputElems = static_cast<size_t>(rows) * static_cast<size_t>(experts);
    const size_t outputElems = static_cast<size_t>(rows) * static_cast<size_t>(k);
    const size_t inputBytes = inputElems * sizeof(float);
    const size_t outputBytes = outputElems * sizeof(float);

    float* inputHost = nullptr;
    float* outputHost = nullptr;
    uint8_t* inputDevice = nullptr;
    uint8_t* outputDevice = nullptr;

    bool ok = CheckAcl(
        "aclrtMallocHost(input)",
        aclrtMallocHost(reinterpret_cast<void**>(&inputHost), inputBytes));
    if (ok) {
        ok = CheckAcl(
            "aclrtMallocHost(output)",
            aclrtMallocHost(reinterpret_cast<void**>(&outputHost), outputBytes));
    }
    if (ok) {
        ok = CheckAcl(
            "aclrtMalloc(input)",
            aclrtMalloc(reinterpret_cast<void**>(&inputDevice), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (ok) {
        ok = CheckAcl(
            "aclrtMalloc(output)",
            aclrtMalloc(reinterpret_cast<void**>(&outputDevice), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    if (ok) {
        FillInput(inputHost, rows, experts, k);
        std::fill(outputHost, outputHost + outputElems, 0.0f);
        ok = CheckAcl(
            "aclrtMemcpy(H2D)",
            aclrtMemcpy(
                inputDevice,
                inputBytes,
                inputHost,
                inputBytes,
                ACL_MEMCPY_HOST_TO_DEVICE));
    }

    const int32_t inner = (experts + 31) / 32 * 32;
    const int32_t blockNum = rows;
    std::printf("RUN profile-one rows=%d E=%d k=%d inner=%d\n", rows, experts, k, inner);
    std::fflush(stdout);

    if (ok) {
#ifdef PROFILE_PIVOT
        sqrt_softplus_pivot_top_k_custom_do(
            static_cast<uint32_t>(blockNum),
            stream,
            inputDevice,
            outputDevice,
            rows,
            experts,
            inner,
            k,
            blockNum);
#else
        sqrt_softplus_prefix_merge_top_k_dispatch_custom_do(
            static_cast<uint32_t>(blockNum),
            stream,
            inputDevice,
            outputDevice,
            rows,
            experts,
            inner,
            k,
            blockNum);
#endif
        ok = CheckAcl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    }

    if (ok) {
        ok = CheckAcl(
            "aclrtMemcpy(D2H)",
            aclrtMemcpy(
                outputHost,
                outputBytes,
                outputDevice,
                outputBytes,
                ACL_MEMCPY_DEVICE_TO_HOST));
    }

    if (ok) {
        for (int32_t row = 0; row < rows && ok; ++row) {
            const size_t inputBase = static_cast<size_t>(row) * static_cast<size_t>(experts);
            std::vector<float> sorted(
                inputHost + inputBase,
                inputHost + inputBase + static_cast<size_t>(experts));
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());

            const size_t outputBase = static_cast<size_t>(row) * static_cast<size_t>(k);
            for (int32_t i = 0; i < k; ++i) {
                const float expected = ReferenceTransform(sorted[static_cast<size_t>(i)]);
                const float actual = outputHost[outputBase + static_cast<size_t>(i)];
                if (std::fabs(actual - expected) > 1.0e-5f) {
                    std::fprintf(
                        stderr,
                        "[MISMATCH] row=%d i=%d actual=%g expected=%g\n",
                        row,
                        i,
                        actual,
                        expected);
                    ok = false;
                    break;
                }
            }
        }
    }

    if (outputDevice != nullptr) {
        (void)aclrtFree(outputDevice);
    }
    if (inputDevice != nullptr) {
        (void)aclrtFree(inputDevice);
    }
    if (outputHost != nullptr) {
        (void)aclrtFreeHost(outputHost);
    }
    if (inputHost != nullptr) {
        (void)aclrtFreeHost(inputHost);
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "用法: %s rows E k\n", argv[0]);
        return 2;
    }

    const int32_t rows = std::atoi(argv[1]);
    const int32_t experts = std::atoi(argv[2]);
    const int32_t k = std::atoi(argv[3]);
    if (rows <= 0 || rows > 64 || experts < 64 || experts > 1024 ||
        k < 2 || k > 8 || k > experts) {
        std::fprintf(stderr, "参数范围: 1<=rows<=64, 64<=E<=1024, 2<=k<=8\n");
        return 2;
    }

    if (!CheckAcl("aclInit", aclInit(nullptr))) {
        return 1;
    }
    bool deviceSet = CheckAcl("aclrtSetDevice", aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    bool ok = deviceSet && CheckAcl("aclrtCreateStream", &stream ? aclrtCreateStream(&stream) : ACL_ERROR_INVALID_PARAM);

    if (ok) {
        ok = RunOne(stream, rows, experts, k);
    }

    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(0);
    }
    (void)aclFinalize();

    if (ok) {
        std::puts("PASS: single-shape profile driver");
    }
    return ok ? 0 : 1;
}
