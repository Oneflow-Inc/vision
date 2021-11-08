#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferNmsTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = Shape({ctx->InputShape("in", 0).At(0)});
  return Maybe<void>::Ok();
}

Maybe<void> InferNmsDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kInt8;
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("nms")
    .Input("in")
    .Output("out")
    .Attr<float>("iou_threshold")
    .Attr<int32_t>("keep_n")
    .SetTensorDescInferFn(InferNmsTensorDesc)
    .SetDataTypeInferFn(InferNmsDataType)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

}  // namespace oneflow
