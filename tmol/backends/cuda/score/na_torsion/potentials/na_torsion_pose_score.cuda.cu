#include <tmol/utility/tensor/native_cuda.hh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

template <typename Real, bool ComputeDerivatives>
__global__ void na_torsion_kernel(
    Real const* coords,
    int64_t const* base,
    bool const* is_na,
    int64_t const* torsion_indices,
    bool const* torsion_ok,
    int64_t const* ring_indices,
    bool const* ring_ok,
    int64_t const* prev,
    Real const* backbone_means,
    Real const* backbone_sdev,
    Real const* sugar_means,
    Real const* chi_means,
    Real const* sdev_sugar,
    Real const* sdev_chi,
    Real const* well_pucker,
    Real const* well_alpha_gamma,
    Real const* well_bibii_pucker,
    Real const* well_alphanext_bibii,
    Real const* well_chi_syn,
    bool const* is_north,
    Real const* weight_bb,
    Real const* weight_chi,
    Real const* weight_sugar,
    Real pucker_temperature,
    Real bin_blend_sdev,
    int n_poses,
    int max_n_blocks,
    int n_atoms,
    int n_bases,
    Real* derivatives,
    Real* output) {
  na_torsion_block<Real, tmol::Device::CUDA, ComputeDerivatives>(
      blockIdx.x * blockDim.x + threadIdx.x,
      coords,
      base,
      is_na,
      torsion_indices,
      torsion_ok,
      ring_indices,
      ring_ok,
      prev,
      backbone_means,
      backbone_sdev,
      sugar_means,
      chi_means,
      sdev_sugar,
      sdev_chi,
      well_pucker,
      well_alpha_gamma,
      well_bibii_pucker,
      well_alphanext_bibii,
      well_chi_syn,
      is_north,
      weight_bb,
      weight_chi,
      weight_sugar,
      pucker_temperature,
      bin_blend_sdev,
      n_poses,
      max_n_blocks,
      n_atoms,
      n_bases,
      derivatives,
      output);
}

std::tuple<tmol::native::Tensor, tmol::native::Tensor>
na_torsion_pose_score_cuda(
    tmol::native::Tensor coords,
    tmol::native::Tensor base,
    tmol::native::Tensor is_na,
    tmol::native::Tensor torsion_indices,
    tmol::native::Tensor torsion_ok,
    tmol::native::Tensor ring_indices,
    tmol::native::Tensor ring_ok,
    tmol::native::Tensor prev,
    tmol::native::Tensor backbone_means,
    tmol::native::Tensor backbone_sdev,
    tmol::native::Tensor sugar_means,
    tmol::native::Tensor chi_means,
    tmol::native::Tensor sdev_sugar,
    tmol::native::Tensor sdev_chi,
    tmol::native::Tensor well_pucker,
    tmol::native::Tensor well_alpha_gamma,
    tmol::native::Tensor well_bibii_pucker,
    tmol::native::Tensor well_alphanext_bibii,
    tmol::native::Tensor well_chi_syn,
    tmol::native::Tensor is_north,
    tmol::native::Tensor weight_bb,
    tmol::native::Tensor weight_chi,
    tmol::native::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs) {
  TORCH_CHECK(coords.is_cuda(), "na_torsion_pose_score requires CUDA tensors");
  TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
  int n_bases = int(chi_means.size(0));
  TORCH_CHECK(
      n_bases > 0 && n_bases % 2 == 0,
      "chi tables must have equal DNA and RNA base counts");
  TORCH_CHECK(
      chi_means.size(1) == N_PUCKER && well_chi_syn.size(2) == n_bases,
      "inconsistent nucleic-acid base table shapes");
  int n_poses = int(base.size(0));
  int max_n_blocks = int(base.size(1));
  auto output = tmol::native::zeros({2, n_poses}, coords.options());
  auto derivatives =
      compute_derivs
          ? tmol::native::zeros({2, coords.size(0), 3}, coords.options())
          : tmol::native::empty({0}, coords.options());
  native::CUDAGuard guard(coords.device());
  cudaStream_t stream = native::current_cuda_stream();
  int threads = 128;
  int blocks = (n_poses * max_n_blocks + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      coords.scalar_type(), "na_torsion_pose_score_cuda", [&] {
        if (!compute_derivs) {
          na_torsion_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
              coords.const_data_ptr<scalar_t>(),
              base.const_data_ptr<int64_t>(),
              is_na.const_data_ptr<bool>(),
              torsion_indices.const_data_ptr<int64_t>(),
              torsion_ok.const_data_ptr<bool>(),
              ring_indices.const_data_ptr<int64_t>(),
              ring_ok.const_data_ptr<bool>(),
              prev.const_data_ptr<int64_t>(),
              backbone_means.const_data_ptr<scalar_t>(),
              backbone_sdev.const_data_ptr<scalar_t>(),
              sugar_means.const_data_ptr<scalar_t>(),
              chi_means.const_data_ptr<scalar_t>(),
              sdev_sugar.const_data_ptr<scalar_t>(),
              sdev_chi.const_data_ptr<scalar_t>(),
              well_pucker.const_data_ptr<scalar_t>(),
              well_alpha_gamma.const_data_ptr<scalar_t>(),
              well_bibii_pucker.const_data_ptr<scalar_t>(),
              well_alphanext_bibii.const_data_ptr<scalar_t>(),
              well_chi_syn.const_data_ptr<scalar_t>(),
              is_north.const_data_ptr<bool>(),
              weight_bb.const_data_ptr<scalar_t>(),
              weight_chi.const_data_ptr<scalar_t>(),
              weight_sugar.const_data_ptr<scalar_t>(),
              scalar_t(pucker_temperature),
              scalar_t(bin_blend_sdev),
              n_poses,
              max_n_blocks,
              int(coords.size(0)),
              n_bases,
              nullptr,
              output.data_ptr<scalar_t>());
        } else {
          na_torsion_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
              coords.const_data_ptr<scalar_t>(),
              base.const_data_ptr<int64_t>(),
              is_na.const_data_ptr<bool>(),
              torsion_indices.const_data_ptr<int64_t>(),
              torsion_ok.const_data_ptr<bool>(),
              ring_indices.const_data_ptr<int64_t>(),
              ring_ok.const_data_ptr<bool>(),
              prev.const_data_ptr<int64_t>(),
              backbone_means.const_data_ptr<scalar_t>(),
              backbone_sdev.const_data_ptr<scalar_t>(),
              sugar_means.const_data_ptr<scalar_t>(),
              chi_means.const_data_ptr<scalar_t>(),
              sdev_sugar.const_data_ptr<scalar_t>(),
              sdev_chi.const_data_ptr<scalar_t>(),
              well_pucker.const_data_ptr<scalar_t>(),
              well_alpha_gamma.const_data_ptr<scalar_t>(),
              well_bibii_pucker.const_data_ptr<scalar_t>(),
              well_alphanext_bibii.const_data_ptr<scalar_t>(),
              well_chi_syn.const_data_ptr<scalar_t>(),
              is_north.const_data_ptr<bool>(),
              weight_bb.const_data_ptr<scalar_t>(),
              weight_chi.const_data_ptr<scalar_t>(),
              weight_sugar.const_data_ptr<scalar_t>(),
              scalar_t(pucker_temperature),
              scalar_t(bin_blend_sdev),
              n_poses,
              max_n_blocks,
              int(coords.size(0)),
              n_bases,
              derivatives.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>());
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, derivatives};
}

template <typename Real>
__global__ void na_torsion_backward_kernel(
    Real const* derivatives,
    Real const* grad_output,
    int64_t grad_stride_score,
    int64_t grad_stride_pose,
    int n_poses,
    int n_atoms,
    Real* grad_coords) {
  na_torsion_weighted_derivative<Real>(
      blockIdx.x * blockDim.x + threadIdx.x,
      derivatives,
      grad_output,
      grad_stride_score,
      grad_stride_pose,
      n_poses,
      n_atoms,
      grad_coords);
}

tmol::native::Tensor na_torsion_pose_score_backward_cuda(
    tmol::native::Tensor derivatives, tmol::native::Tensor grad_output) {
  TORCH_CHECK(derivatives.is_cuda(), "NA torsion derivatives must be CUDA");
  TORCH_CHECK(
      grad_output.is_cuda(), "NA torsion output gradients must be CUDA");
  TORCH_CHECK(
      derivatives.dim() == 3 && derivatives.size(0) == 2,
      "expected derivatives shaped [2, n_atoms, 3]");
  TORCH_CHECK(
      grad_output.dim() == 2 && grad_output.size(0) == 2,
      "expected output gradients shaped [2, n_poses]");
  int n_poses = int(grad_output.size(1));
  int n_atoms = int(derivatives.size(1));
  TORCH_CHECK(n_atoms % n_poses == 0, "atoms must divide evenly among poses");
  auto grad_coords = tmol::native::empty({n_atoms, 3}, derivatives.options());
  native::CUDAGuard guard(derivatives.device());
  cudaStream_t stream = native::current_cuda_stream();
  int threads = 256;
  int blocks = (n_atoms * 3 + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      derivatives.scalar_type(), "na_torsion_pose_score_backward_cuda", [&] {
        na_torsion_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            derivatives.const_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            grad_output.stride(0),
            grad_output.stride(1),
            n_poses,
            n_atoms,
            grad_coords.data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_coords;
}

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
