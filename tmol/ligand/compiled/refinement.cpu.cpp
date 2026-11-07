// One host ABI compiled into either framework's native library.
#include <tmol/ligand/compiled/refinement.hh>
#include <tmol/utility/portable/denormals.hh>
#include <type_traits>

#if defined(_WIN32)
#define TMOL_LIGAND_EXPORT __declspec(dllexport)
#else
#define TMOL_LIGAND_EXPORT __attribute__((visibility("default")))
#endif

using tmol::ligand::RefinementInput;
using tmol::ligand::RefinementStatus;
static_assert(std::is_standard_layout<RefinementInput<float>>::value);
static_assert(std::is_standard_layout<RefinementInput<double>>::value);

extern "C" {
TMOL_LIGAND_EXPORT int tmol_ligand_refinement_schema_version() { return 1; }
TMOL_LIGAND_EXPORT int tmol_ligand_refinement_f32(
    const RefinementInput<float>* input, float* value, float* gradient) {
  if (!input || !value) return int(RefinementStatus::invalid_shape);
  return int(tmol::ligand::refinement_value_gradient(*input, *value, gradient));
}
TMOL_LIGAND_EXPORT int tmol_ligand_refinement_f64(
    const RefinementInput<double>* input, double* value, double* gradient) {
  if (!input || !value) return int(RefinementStatus::invalid_shape);
  return int(tmol::ligand::refinement_value_gradient(*input, *value, gradient));
}
TMOL_LIGAND_EXPORT std::int64_t tmol_ligand_refinement_size_f32() {
  return sizeof(RefinementInput<float>);
}
TMOL_LIGAND_EXPORT std::int64_t tmol_ligand_refinement_size_f64() {
  return sizeof(RefinementInput<double>);
}
// -1 denotes unsupported thread-local denormal control.
TMOL_LIGAND_EXPORT int tmol_ligand_refinement_denormal_mode() {
  return tmol::utility::denormal_mode();
}
// Status: 0 success, 1 invalid mode, 2 unsupported architecture/mode.
TMOL_LIGAND_EXPORT int tmol_ligand_refinement_set_denormal_mode(int mode) {
  if (mode < 0 || mode > 3) return 1;
  return tmol::utility::set_denormal_mode(mode) ? 0 : 2;
}
}
