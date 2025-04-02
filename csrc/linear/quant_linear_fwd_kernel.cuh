// The code is adapted from https://github.com/vllm-project/vllm/blob/7a9cb294ae317b28a60165b34c8398c762869a74/csrc/quantization/cutlass_w8a8/scaled_mm_dq_c2x.cu#L157

#pragma once

#include <stddef.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include "broadcast_load_epilogue_c2x.hpp"

// // C++ interface
// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

namespace {

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages>
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  // static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  // static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                cutlass::arch::OpMultiplyAdd>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute1>;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType = 
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, 16, 
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, 16, 
      float, RowMajor, 4,
      ElementAcc, float, cutlass::arch::OpClassTensorOp, 
      Arch, 
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};


template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_>
struct cutlass_2x_gemv {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;

  int const kElementsPerAccess = 16;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementD,   // output type
      1,          // number of elements computed per operation, always use 1 for GEMV
      ElementAcc, // multiply type
      float       // compute type
      >; 

  using KernelType = 
    ArchGuard<typename cutlass::gemm::kernel::Gemv<
          ElementAB,           // Element A
          RowMajor,            // Layout A
          ElementAB,           // Element B
          ElementD,            // Element C
          ElementAcc,          // Element accumulator
          EpilogueOp,          // Output operator
          128 / cutlass::sizeof_bits<ElementD>::value      // Element access granularity
          >>;
  // clang-format on
  using Op = cutlass::gemm::device::Gemv<KernelType>;
};

template <typename Gemm>
void cutlass_scaled_mm_dq_dispatcher(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
  StrideC c_stride{ldc, cute::Int<1>{}, cute::Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  auto a_scales_ptr = a_scales.data_ptr<float>();
  auto b_scales_ptr = b_scales.data_ptr<float>();

  using ScaleAArgs = typename Gemm::ScaleA::Arguments;
  using ScaleBArgs = typename Gemm::ScaleB::Arguments;

  ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
  ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};

  typename Gemm::EVTCompute0::Arguments evt0_compute_args{b_args};

  typename Gemm::EVTCompute1::Arguments evt1_compute_args{a_args,
                                                          evt0_compute_args};
  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  typename Gemm::EVTD::Arguments epilogue_args{
      evt1_compute_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,  // universal mode
      problem_size,                                           // problem size
      1,                                                      // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  // Launch the CUTLASS GEMM kernel.
  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}


template <typename Gemv>
void cutlass_scaled_mv_dq_dispatcher(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     float const& alpha,
                                     float const& beta) {
  using ElementAB = typename Gemv::ElementAB;
  using LayoutInputA = typename Gemv::LayoutInputA;
  using ElementD = typename Gemv::ElementD;

  int32_t bsize = b.size(0); // x
  int32_t m = a.size(0);     // w
  int32_t k = a.size(1);     // w
  cutlass::MatrixCoord problem_size{m, k};
 
  auto a_size = cutlass::MatrixCoord(bsize*m, k);
  auto b_size = cutlass::MatrixCoord(bsize*k, 1);
  auto out_size = cutlass::MatrixCoord(bsize*m, 1);

  cutlass::TensorRef<ElementAB, LayoutInputA> a_ref(
      a.data_ptr<int8_t>(), LayoutInputA::packed(a_size));
  auto b_ptr = static_cast<int8_t*>(b.data_ptr());
  auto c_ptr = static_cast<int8_t*>(out.data_ptr());

  typename Gemv::Op::Arguments args{problem_size,  // <- problem size
                                    bsize,         // <- batch size
                                    {alpha, beta}, // <- tuple of alpha and beta
                                    a_ref,  // <- reference to matrix A on device
                                    b_ptr,  // <- reference to matrix B on device
                                    c_ptr,  // <- reference to matrix C on device
                                    c_ptr,  // <- reference to matrix D on device
                                    m*k,
                                    k,
                                    m,
                                    m};


  // Launch the CUTLASS GEMM kernel.
  typename Gemv::Op gemv_op;
  size_t workspace_size = gemv_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemv_op.can_implement(args));
  cutlass::Status status = gemv_op(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace


void cutlass_scaled_mm_dq_sm80(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_scaled_mm_dq_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, cutlass::bfloat16_t,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                    b_scales);
  } else if (out.dtype() == torch::kFloat16) {
    return cutlass_scaled_mm_dq_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, cutlass::half_t,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                    b_scales);
  } else if (out.dtype() == torch::kInt8) {
    return cutlass_scaled_mm_dq_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, int8_t,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                    b_scales);
  } else {
    throw std::invalid_argument("unsupported output type");
  }
}


void cutlass_scaled_mv_dq_sm80(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               float const& alpha,
                               float const& beta) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  if (out.dtype() == torch::kBFloat16) {
    return cutlass_scaled_mv_dq_dispatcher<cutlass_2x_gemv<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, cutlass::bfloat16_t>>(
          out, a, b, alpha, beta);
  } else if (out.dtype() == torch::kFloat16) {
    return cutlass_scaled_mv_dq_dispatcher<cutlass_2x_gemv<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, cutlass::half_t>>(
          out, a, b, alpha, beta);
  } else if (out.dtype() == torch::kInt8) {
    return cutlass_scaled_mv_dq_dispatcher<cutlass_2x_gemv<
        cutlass::arch::Sm80, enable_sm80_to_sm89, int8_t, int8_t>>(
          out, a, b, alpha, beta);
  } else {
    throw std::invalid_argument("unsupported output type");
  }
}

void cutlass_scaled_mm_dq(torch::Tensor& c, torch::Tensor const& a,
                          torch::Tensor const& b, torch::Tensor const& a_scales,
                          torch::Tensor const& b_scales) {

  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  // Ampere
  cutlass_scaled_mm_dq_sm80(c, a, b, a_scales, b_scales);
}


void cutlass_scaled_mv_dq(torch::Tensor& c, torch::Tensor const& a,
                          torch::Tensor const& b, float const& alpha,
                          float const& beta) {

  // Checks for conformality
  // a: w, b: x, c: y => y = wx
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == b.size(0) && a.size(1) == b.size(1) &&
              a.size(0) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(1) == 1);                      // Row-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(0) % 16 == 0);  // 16 Byte Alignment

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  // Ampere
  cutlass_scaled_mv_dq_sm80(c, a, b, alpha, beta);
}
