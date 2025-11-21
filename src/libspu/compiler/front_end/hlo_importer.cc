// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/compiler/front_end/hlo_importer.h"

#include <iostream>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/convolution_4d_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/real_imag_expander.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/batch_dot_simplification.h"
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_module_importer.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/map_inliner.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape_util.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"

#include "xla/service/hlo.pb.h"

namespace xla {

class GatherSimplifierOwn : public OpExpanderPass {
public:
  absl::string_view name() const override { return "gather_simplifier"; }

  static bool IsSimplifiedGather(const HloGatherInstruction *gather);

protected:
  bool InstructionMatchesPattern(HloInstruction *inst) override;

  absl::StatusOr<HloInstruction *>
  ExpandInstruction(HloInstruction *inst) override;
};

absl::StatusOr<HloInstruction *>
GatherSimplifierOwn::ExpandInstruction(HloInstruction *inst) {
  auto *gather = DynCast<HloGatherInstruction>(inst);

  // If any slice size is 0, we can just return a constant zero.
  if (absl::c_linear_search(gather->gather_slice_sizes(), 0)) {
    auto *zero = gather->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(gather->shape().element_type())));
    return gather->AddInstruction(
        HloInstruction::CreateBroadcast(gather->shape(), zero, {}));
  }

  const auto &dims = gather->gather_dimension_numbers();
  int operand_rank =
      dims.collapsed_slice_dims().size() + dims.offset_dims().size();

  // Make the operand conform to start_index_map.
  auto [operand_permutation, operand_permutation_inverse] =
      MakeOperandStartIndexPermutations(dims.start_index_map(), operand_rank);
  auto *operand = gather->operands()[0];
  auto *start_indices = gather->operands()[1];
  TF_ASSIGN_OR_RETURN(operand, MaybeTranspose(operand, operand_permutation));
  TF_ASSIGN_OR_RETURN(
      start_indices,
      TransformStartIndices(start_indices, dims.index_vector_dim()));

  // Permute the slice sizes according to start_index_map and compute the new
  // output shape for the Gather op.
  auto slice_sizes = Permute(gather->gather_slice_sizes(), operand_permutation);
  std::vector<int64_t> output_dims = {start_indices->shape().dimensions(0)};
  absl::c_copy(slice_sizes, std::back_inserter(output_dims));
  Shape output_shape =
      ShapeUtil::MakeShape(operand->shape().element_type(), output_dims);

  std::vector<int64_t> offset_dims(operand_rank);
  absl::c_iota(offset_dims, 1);
  std::vector<int64_t> start_index_map(dims.start_index_map().size());
  absl::c_iota(start_index_map, 0);

  auto *result = gather->AddInstruction(HloInstruction::CreateGather(
      output_shape, operand, start_indices,
      HloGatherInstruction::MakeGatherDimNumbers(
          offset_dims,
          /*collapsed_slice_dims=*/{}, start_index_map, /*index_vector_dim=*/1),
      slice_sizes, gather->indices_are_sorted()));

  // Undo the start_index_map transpose.
  std::vector<int64_t> output_permutation(1 + // start index dimension.
                                          operand_rank);
  absl::c_transform(operand_permutation_inverse, output_permutation.begin() + 1,
                    [](int64_t dim) { return dim + 1; });
  TF_ASSIGN_OR_RETURN(result, MaybeTranspose(result, output_permutation));

  // Collapse the requested slice dimensions.
  if (!dims.collapsed_slice_dims().empty()) {
    std::vector<int64_t> collapsed_slice_dims(
        dims.collapsed_slice_dims().size());
    absl::c_transform(dims.collapsed_slice_dims(), collapsed_slice_dims.begin(),
                      [](int64_t dim) { return dim + 1; });
    TF_ASSIGN_OR_RETURN(result,
                        ElideDegenerateDims(result, collapsed_slice_dims));
  }

  // Expand the start index dimensions.
  auto original_start_index_dims = gather->operands()[1]->shape().dimensions();
  std::vector<int64_t> start_indices_dims;
  for (int i = 0; i < original_start_index_dims.size(); ++i) {
    if (i != dims.index_vector_dim()) {
      start_indices_dims.push_back(original_start_index_dims[i]);
    }
  }
  if (start_indices_dims.size() > 1) {
    TF_ASSIGN_OR_RETURN(result,
                        ExpandFirstDimIntoNDims(result, start_indices_dims));
  } else if (start_indices_dims.empty()) {
    TF_ASSIGN_OR_RETURN(result, ElideDegenerateDims(result, {0}));
  }

  // Move the offset dims to the final locations.
  std::vector<int64_t> output_perm;
  auto output_rank = static_cast<int64_t>(start_indices_dims.size() +
                                          dims.offset_dims().size());
  output_perm.reserve(output_rank);
  auto offset_dim_index = static_cast<int64_t>(start_indices_dims.size());
  int64_t start_index_dim_index = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (absl::c_linear_search(dims.offset_dims(), i)) {
      output_perm.push_back(offset_dim_index++);
    } else {
      output_perm.push_back(start_index_dim_index++);
    }
  }
  return MaybeTranspose(result, output_perm);
}

bool GatherSimplifierOwn::IsSimplifiedGather(
    const HloGatherInstruction *gather) {
  auto *start_indices = gather->operands()[1];
  const auto &dims = gather->gather_dimension_numbers();
  return start_indices->shape().dimensions().size() == 2 &&
         dims.index_vector_dim() == 1 &&
         IsIdentityPermutation(dims.start_index_map()) &&
         dims.collapsed_slice_dims().empty() &&
         *dims.offset_dims().begin() == 1 &&
         *dims.offset_dims().rbegin() == dims.offset_dims().size();
}

bool GatherSimplifierOwn::InstructionMatchesPattern(HloInstruction *inst) {
  auto *gather = DynCast<HloGatherInstruction>(inst);
  // ------------------- [ SPU PATCH - BEGIN ] -------------------
  //
  // 原始代码 (auto* gather = DynCast<HloGatherInstruction>(inst);)
  //        (return gather && !IsSimplifiedGather(gather);)
  //
  // 存在一个 Bug：GatherSimplifier 无法正确处理 batched gather
  // (即 HLO 中带有 operand_batching_dims 或
  //  start_indices_batching_dims 的 gather)。
  //
  // ExpandInstruction 中的逻辑（例如 operand_rank 的计算和
  // start_index_map 的转换）在 batched gather 上会做出
  // 错误的假设，导致其生成非法的 HLO（例如无效的 transpose）。
  //
  // 修复：明确检查是否为 batched gather，如果是，则跳过此 Pass。
  //
  if (!gather) {
    return false;
  }

  const auto &dims = gather->gather_dimension_numbers();
  if (!dims.operand_batching_dims().empty() ||
      !dims.start_indices_batching_dims().empty()) {
    // 这是一个 batched gather，GatherSimplifier 不支持它。
    // 返回 false 以跳过。
    return false;
  }
  // -------------------- [ SPU PATCH - END ] --------------------

  // 如果它不是 batched gather，则应用原始逻辑
  return !IsSimplifiedGather(gather);
}

void runHloPasses(xla::HloModule *module) {

  // Simplifier options
  AlgebraicSimplifierOptions options;
  // For MPC, dot is way faster than reduce
  options.set_enable_dot_strength_reduction(false);
  // We do not handle nan, so just use faster minmax
  options.set_minmax_propagate_nan(false);
  // Transpose and reshape is cheep for us
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  HloPassPipeline pipeline("optimization");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);

  pipeline.AddPass<OperandUpcaster>();
  pipeline.AddPass<ResultCaster>();

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  pipeline.AddPass<ConditionalToSelect>();
  pipeline.AddPass<MapInliner>();

  pipeline.AddPass<CholeskyExpander>(); // Eliminate chol
  pipeline.AddPass<QrExpander>();       // Eliminate qr
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();

  // Convert BF16 operations to F32 operations so that the SPU backend can
  // support BF16 operations without directly implementing a BF16 lowering for
  // most ops.
  FloatSupport bf16_support(BF16);
  pipeline.AddPass<FloatNormalization>(&bf16_support);

  // Inline computations with a single call site.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<gpu::DotDimensionSorter>();
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>(); // Simplify dot

  pipeline.AddPass<Convolution4DExpander>();

  // After canonicalization, there may be more batch dots that can be
  // simplified.
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand*/ [](HloInstruction *) { return true; },
      /*is_cost_viable*/ [](HloInstruction *) { return true; },
      /*convert_batch_groups_only=*/false);
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);

  // Run the following passes to a fixed point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    pipeline.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/false,
        /*allow_mixed_precision=*/false);

    pipeline.AddPass<GatherSimplifierOwn>();
    pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<BitcastDtypesExpander>();
    // AlgebraicSimplifier may add contracting dimensions to a dot.
    pipeline.AddPass<gpu::DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();

    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<SliceSinker>();
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<RealImagExpander>();
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
  }();

  auto status = pipeline.Run(module).status();

  SPU_ENFORCE(status.ok());
}
} // namespace xla

namespace spu::compiler {

mlir::OwningOpRef<mlir::ModuleOp>
HloImporter::parseXlaModuleFromString(const std::string &content) {
  // Stage 1: Load hlo_module
  xla::HloModuleProto hlo_module;
  if (!hlo_module.ParseFromString(content)) {
    // If parse as HloModuleProto fails, try HloProto.
    xla::HloProto hlo_proto;
    if (!hlo_proto.ParseFromString(content)) {
      // Try human-readable format
      if (!google::protobuf::TextFormat::ParseFromString(content, &hlo_proto)) {
        SPU_THROW("Failed to parse hlo module from string {}", content);
      }
    }
    hlo_module = hlo_proto.hlo_module();
  }

  xla::DebugOptions debug_options;

  if (context_->hasPrettyPrintEnabled()) {
    debug_options.set_xla_dump_hlo_pass_re(".*");
    debug_options.set_xla_dump_to(context_->getPrettyPrintDir().string());
    switch (context_->getXlaPrettyPrintKind()) {
    case spu::XLAPrettyPrintKind::DOT: {
      debug_options.set_xla_dump_hlo_as_dot(true);
      break;
    }
    case spu::XLAPrettyPrintKind::HTML: {
      debug_options.set_xla_dump_hlo_as_html(true);
      break;
    }
    default: {
      debug_options.set_xla_dump_hlo_as_text(true);
      break;
    }
    }
    debug_options.set_xla_enable_dumping(true);
  }

  auto module_config =
      xla::HloModule::CreateModuleConfigFromProto(hlo_module, debug_options);
  if (!module_config.status().ok()) {
    SPU_THROW("{}", module_config.status().message());
  }

  auto module = xla::HloModule::CreateFromProto(hlo_module, *module_config);
  if (!module.status().ok()) {
    SPU_THROW("{}", module.status().message());
  }

  xla::runHloPasses((*module).get());

  // Stage 2: Ask mlir hlo to convert xla module into mlir
  // Create importer
  auto mlir_hlo = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(
      mlir::UnknownLoc::get(context_->getMLIRContext())));
  xla::HloModuleImporter importer(mlir_hlo.get());

  auto status = importer.Import(**module);
  if (!status.ok()) {
    SPU_THROW("{}", status.message());
  }

  return mlir_hlo;
}

} // namespace spu::compiler
