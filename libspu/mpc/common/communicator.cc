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

#include "libspu/mpc/common/communicator.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

// When ArrayRef is transferred via network, it's supposed to be compact.
constexpr int64_t kOffset = 0;

std::shared_ptr<yacl::Buffer> stealBuffer(yacl::Buffer&& buf) {
  return std::make_shared<yacl::Buffer>(std::move(buf));
}

}  // namespace

NdArrayRef Communicator::allReduce(ReduceOp op, const NdArrayRef& in,
                                   std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<yacl::Buffer> bufs = yacl::link::AllGather(lctx_, *buf, tag);

  SPU_ENFORCE(bufs.size() == getWorldSize());
  auto res = in.clone();
  for (size_t idx = 0; idx < bufs.size(); idx++) {
    if (idx == getRank()) {
      continue;
    }

    auto arr = NdArrayRef(stealBuffer(std::move(bufs[idx])), in.eltype(),
                          in.shape(), makeCompactStrides(in.shape()), kOffset);
    if (op == ReduceOp::ADD) {
      ring_add_(res, arr);
    } else if (op == ReduceOp::XOR) {
      ring_xor_(res, arr);
    } else {
      SPU_THROW("unsupported reduce op={}", static_cast<int>(op));
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size() * (lctx_->WorldSize() - 1);

  return res;
}

NdArrayRef Communicator::reduce(ReduceOp op, const NdArrayRef& in, size_t root,
                                std::string_view tag) {
  SPU_ENFORCE(root < lctx_->WorldSize());
  const auto buf = in.getOrCreateCompactBuf();

  std::vector<yacl::Buffer> bufs = yacl::link::Gather(lctx_, *buf, root, tag);

  auto res = in.clone();
  if (getRank() == root) {
    for (size_t idx = 0; idx < bufs.size(); idx++) {
      if (idx == getRank()) {
        continue;
      }

      auto arr =
          NdArrayRef(stealBuffer(std::move(bufs[idx])), in.eltype(), in.shape(),
                     makeCompactStrides(in.shape()), kOffset);
      if (op == ReduceOp::ADD) {
        ring_add_(res, arr);
      } else if (op == ReduceOp::XOR) {
        ring_xor_(res, arr);
      } else {
        SPU_THROW("unsupported reduce op={}", static_cast<int>(op));
      }
    }
  }

  stats_.latency += 1;
  stats_.comm += buf->size();

  return res;
}

NdArrayRef Communicator::rotate(const NdArrayRef& in, std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  lctx_->SendAsync(lctx_->PrevRank(), *buf, tag);

  auto res_buf = lctx_->Recv(lctx_->NextRank(), tag);

  stats_.latency += 1;
  stats_.comm += buf->size();

  return NdArrayRef(stealBuffer(std::move(res_buf)), in.eltype(), in.shape(),
                    makeCompactStrides(in.shape()), kOffset);
}

void Communicator::sendAsync(size_t dst_rank, const NdArrayRef& in,
                             std::string_view tag) {
  const auto buf = in.getOrCreateCompactBuf();

  lctx_->SendAsync(dst_rank, *buf, tag);
}

NdArrayRef Communicator::recv(size_t src_rank, const Type& eltype,
                              std::string_view tag) {
  auto buf = lctx_->Recv(src_rank, tag);

  int64_t numel = buf.size() / eltype.size();
  return NdArrayRef(stealBuffer(std::move(buf)), eltype, {numel}, {1}, kOffset);
}

}  // namespace spu::mpc
