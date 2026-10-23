//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include "sampling.h"
// EdgeForge adaptation; see NOTICE.md for upstream pin and change list.
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <numeric>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

static void print_devmem_info(bm_handle_t &bm_handle) {
  bm_dev_stat_t stat;
  auto ret = bm_get_stat(bm_handle, &stat);
  if (ret != BM_SUCCESS) {
    std::cerr << "Failed to get device status" << std::endl;
    return;
  }
  std::cout << "DevMem: " << stat.mem_used << "/" << stat.mem_total << " MB"
            << std::endl;
}

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage].output_mems[i]);
  }
}

class Qwen3_VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                   ArrayInt const &pos_idx, ArrayFloat const &pos_weight,
                   ArrayInt const &grid_thw, int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  void clear_history();
  void configure_generation(bool sample, float temp, float p, int k, float penalty) {
    if (!std::isfinite(temp) || temp <= 0 || !std::isfinite(p) || p <= 0 || p > 1 ||
        k < 0 || !std::isfinite(penalty) || penalty <= 0)
      throw std::invalid_argument("Invalid generation parameters");
    do_sample = sample; temperature = temp; top_p = p; top_k = k;
    repetition_penalty = penalty;
  }
  void set_seed(uint32_t seed) { sgen.seed(seed); }
  bool do_sample = true;
  float temperature = 0.7f, top_p = 0.8f, repetition_penalty = 1.0f;
  int top_k = 20;

  std::mt19937 sgen;
  Qwen3_VL() : sgen(std::random_device()()){};
  ~Qwen3_VL() { deinit(); }

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask,
                         int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int select_decode_stage();
  int forward_first_with_kv(ArrayInt const &position_ids);
  int generate(bm_device_mem_t &logits_mem);
  int greedy_search(bm_device_mem_t &logits_mem);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES; // kv bytes for one token
  int NUM_LAYERS;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  bool lmhead_with_topk;
  bool support_history;
  bool is_dynamic;
  bool prefill_mask;
  bool vit_dynamic;
  uint16_t mask_value;
  bool vit_run = false;
  int num_deepstack = 0;
  std::vector<int> visited_tokens;

private:
  bm_handle_t bm_handle = nullptr;
  void *p_bmrt = nullptr;
  bool buffer_allocated = false;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<const bm_net_info_t *> net_blocks_kv;
  std::vector<int> decode_stage_len;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_add;
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> deepstack_buffers;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen3_VL::init_tensors(const bm_net_info_t *net,
                            std::vector<bm_tensor_t> &in_tensors,
                            std::vector<bm_tensor_t> &out_tensors, int stage) {
  in_tensors.resize(net->input_num);
  out_tensors.resize(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage].input_shapes[i]);
  }

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[stage].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage].output_shapes[i]);
  }
}

void Qwen3_VL::net_launch(const bm_net_info_t *net,
                          const std::vector<bm_tensor_t> &in_tensors,
                          std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // BMRT launch is asynchronous. In particular, older SoC runtimes require
  // an explicit fence before CPU sampling or dependent device copies.
  if (bm_thread_sync(bm_handle) != BM_SUCCESS)
    throw std::runtime_error("TPU network synchronization failed");
}

void Qwen3_VL::net_launch_decode(int idx, int kv_offset,
                                 bm_device_mem_t &input_mem, const int *pos_id,
                                 std::vector<uint16_t> &attention_mask,
                                 int stage_idx) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors, stage_idx);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    // position_ids and attention_mask are identical across layers, so reuse the
    // mem filled by layer 0 of the same stage
    in_tensors[1].device_mem =
        net_blocks_cache[0]->stages[stage_idx].input_mems[1];
    in_tensors[2].device_mem =
        net_blocks_cache[0]->stages[stage_idx].input_mems[2];
  }
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void Qwen3_VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                   int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen3_VL::clear_history() {
  // Always reset fresh-call caches, including the no-history profile.
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
  token_length = 0;
  vit_run = false;
}

void Qwen3_VL::init_by_names() {
  auto require_net = [&](const std::string& name, int inputs, int outputs) {
    const auto* net = bmrt_get_network_info(p_bmrt, name.c_str());
    if (!net || net->stage_num != 1 || net->input_num != inputs ||
        net->output_num != outputs || net->is_dynamic)
      throw std::runtime_error("Missing/incompatible static Qwen3-VL graph: " + name);
    return net;
  };
  net_embed = require_net("embedding", 1, 1);
  net_embed_cache = require_net("embedding_cache", 1, 1);
  net_vit = require_net("vit", 5, 4);
  net_lm = require_net("lm_head", 1, 1);
  net_add = require_net("add", 2, 1);
  if (bmrt_get_network_info(p_bmrt, "block_kv_0") != nullptr)
    throw std::runtime_error("EdgeForge profile requires use_block_with_kv=false");
  NUM_LAYERS = 28;
  support_history = false;
  net_blocks.clear();
  net_blocks_cache.clear();
  decode_stage_len.clear();
  for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    net_blocks.push_back(require_net("block_" + std::to_string(layer), 3, 3));
    net_blocks_cache.push_back(require_net("block_cache_" + std::to_string(layer), 5, 3));
  }
  auto shape = [](const bm_shape_t& actual, std::initializer_list<int> expected,
                  const std::string& name) {
    if (actual.num_dims != static_cast<int>(expected.size()) ||
        !std::equal(expected.begin(), expected.end(), actual.dims))
      throw std::runtime_error("Unexpected graph shape: " + name);
  };
  auto dtype = [](bm_data_type_t actual, bm_data_type_t expected, const std::string& name) {
    if (actual != expected) throw std::runtime_error("Unexpected graph dtype: " + name);
  };
  // Validate ranks before reading capacity dimensions. Capacities are compared
  // against the source compilation manifest by Python immediately after init.
  if (net_embed->stages[0].input_shapes[0].num_dims != 2 ||
      net_vit->stages[0].input_shapes[0].num_dims != 2 ||
      net_blocks_cache[0]->stages[0].input_shapes[3].num_dims != 4)
    throw std::runtime_error("Invalid embedding/vision/KV graph ranks");
  const int length = net_embed->stages[0].input_shapes[0].dims[1];
  const int patches = net_vit->stages[0].input_shapes[0].dims[0];
  const int cache_length = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  if (length <= 1 || cache_length != length || patches <= 0 || patches % 4)
    throw std::runtime_error("Invalid prefill/sequence/vision capacities");
  const auto hidden_dtype = net_embed_cache->output_dtypes[0];
  if (hidden_dtype != BM_FLOAT16 && hidden_dtype != BM_BFLOAT16)
    throw std::runtime_error("Require fp16 or bf16 hidden states");
  for (auto pair : {std::make_pair(net_embed, length), std::make_pair(net_embed_cache, 1)}) {
    shape(pair.first->stages[0].input_shapes[0], {1, pair.second}, "embedding input");
    shape(pair.first->stages[0].output_shapes[0], {1, pair.second, 2048}, "embedding output");
    dtype(pair.first->input_dtypes[0], BM_INT32, "embedding tokens");
    dtype(pair.first->output_dtypes[0], hidden_dtype, "embedding hidden");
  }
  shape(net_lm->stages[0].input_shapes[0], {1, 2048}, "lm_head input");
  shape(net_lm->stages[0].output_shapes[0], {1, 151936}, "raw lm_head logits (--do_sample required)");
  dtype(net_lm->input_dtypes[0], hidden_dtype, "lm_head input");
  if (net_lm->output_dtypes[0] != BM_FLOAT32 && net_lm->output_dtypes[0] != BM_FLOAT16 &&
      net_lm->output_dtypes[0] != BM_BFLOAT16)
    throw std::runtime_error("Unsupported raw lm_head logits dtype");
  shape(net_vit->stages[0].input_shapes[0], {patches, 1536}, "ViT pixels");
  shape(net_vit->stages[0].input_shapes[1], {patches, 2}, "ViT rotary positions");
  shape(net_vit->stages[0].input_shapes[2], {patches, 4}, "ViT interpolation indices");
  shape(net_vit->stages[0].input_shapes[3], {patches, 4, 1}, "ViT interpolation weights");
  shape(net_vit->stages[0].input_shapes[4], {1, 1, patches, patches}, "ViT attention mask");
  for (int i = 0; i < 5; ++i)
    dtype(net_vit->input_dtypes[i], (i == 1 || i == 2) ? BM_INT32 : BM_FLOAT32, "ViT input");
  for (int i = 0; i < 4; ++i) {
    shape(net_vit->stages[0].output_shapes[i], {patches / 4, 2048}, "ViT/DeepStack output");
    dtype(net_vit->output_dtypes[i], hidden_dtype, "ViT/DeepStack output");
  }
  for (int i = 0; i < 2; ++i) {
    shape(net_add->stages[0].input_shapes[i], {length * 2048}, "DeepStack add input");
    dtype(net_add->input_dtypes[i], hidden_dtype, "DeepStack add input");
  }
  shape(net_add->stages[0].output_shapes[0], {length * 2048}, "DeepStack add output");
  dtype(net_add->output_dtypes[0], hidden_dtype, "DeepStack add output");
  for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    for (int decode = 0; decode < 2; ++decode) {
      const auto* net = decode ? net_blocks_cache[layer] : net_blocks[layer];
      const int tokens = decode ? 1 : length;
      const auto& stage = net->stages[0];
      shape(stage.input_shapes[0], {1, tokens, 2048}, "block hidden input");
      shape(stage.input_shapes[1], {3, tokens}, "block mRoPE positions");
      shape(stage.input_shapes[2], {1, 1, tokens, decode ? cache_length + 1 : length}, "block attention mask");
      dtype(net->input_dtypes[0], hidden_dtype, "block hidden input");
      dtype(net->input_dtypes[1], BM_INT32, "block mRoPE positions");
      dtype(net->input_dtypes[2], hidden_dtype, "block attention mask");
      shape(stage.output_shapes[0], {1, tokens, 2048}, "block hidden output");
      dtype(net->output_dtypes[0], hidden_dtype, "block hidden output");
      for (int kv = 1; kv <= 2; ++kv) {
        shape(stage.output_shapes[kv], {1, tokens, 8, 128}, "block KV output");
        dtype(net->output_dtypes[kv], hidden_dtype, "block KV output");
        if (decode) {
          shape(stage.input_shapes[kv + 2], {1, cache_length, 8, 128}, "block KV cache");
          dtype(net->input_dtypes[kv + 2], hidden_dtype, "block KV cache");
        }
      }
    }
  }
  decode_stage_len.push_back(cache_length);
  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  if (NUM_LAYERS != 28 || net_vit->output_num != 4 || net_vit->input_num != 5 ||
      net_vit->is_dynamic || net_blocks[0]->is_dynamic)
    throw std::runtime_error("Require static Cosmos 28-layer / four-output DeepStack ViT graphs");
  is_dynamic = net_blocks[0]->is_dynamic;
  prefill_mask = net_blocks[0]->input_num > 2; // with prefill attention mask
  vit_dynamic = net_vit->is_dynamic;
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  if (lmhead_with_topk)
    throw std::runtime_error("Raw lm_head logits required: recompile with --do_sample");
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 16 * 16;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  const auto& kv_shape = net_blocks_cache[0]->stages[0].input_shapes[3];
  if (HIDDEN_SIZE != 2048 || VIT_DIMS != 1536 || kv_shape.num_dims != 4 ||
      kv_shape.dims[0] != 1 || kv_shape.dims[2] != 8 || kv_shape.dims[3] != 128 ||
      net_lm->stages[0].output_shapes[0].dims[1] != 151936)
    throw std::runtime_error("BModel shapes do not match Cosmos-Reason2-2B architecture");
  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 32, 32);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks_kv[0]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void Qwen3_VL::init(int dev_id, std::string model_path) {
  if (p_bmrt || bm_handle) throw std::runtime_error("Runtime already initialized");
  try {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  if (status != BM_SUCCESS) throw std::runtime_error("Cannot open Sophgo device; check device_id and driver");

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  if (!p_bmrt) throw std::runtime_error("Failed to create Sophgo BMRuntime");
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  if (!ret) throw std::runtime_error("BMRuntime failed to load BModel; check compiler/runtime/target compatibility");
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  init_by_names();

  visited_tokens.resize(SEQLEN);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  uint32_t buffer_size = 0;
  if (!support_history) {
    buffer_size = bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  } else {
    // for history, we need a big buffer to store long input
    buffer_size = SEQLEN * HIDDEN_SIZE * sizeof(uint16_t);
  }
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  if (status != BM_SUCCESS) throw std::runtime_error("Insufficient device memory for embedding buffer");
  buffer_allocated = true;
  num_deepstack = net_vit->output_num - 1;
  // assert(num_deepstack > 0);
  for (int i = 0; i < num_deepstack; i++) {
    bm_device_mem_t mem;
    status = bm_malloc_device_byte(bm_handle, &mem, buffer_size);
    if (status != BM_SUCCESS) throw std::runtime_error("Insufficient device memory for DeepStack buffers");
    deepstack_buffers.push_back(mem);
  }
  vit_run = false;
  } catch (...) { deinit(); throw; }
}

void Qwen3_VL::deinit() {
  if (bm_handle) {
    for (auto& buffer : deepstack_buffers) bm_free_device(bm_handle, buffer);
    deepstack_buffers.clear();
    if (buffer_allocated) { bm_free_device(bm_handle, dev_buffer); buffer_allocated = false; }
  }
  if (p_bmrt) { bmrt_destroy(p_bmrt); p_bmrt = nullptr; }
  if (bm_handle) { bm_dev_free(bm_handle); bm_handle = nullptr; }
}

int Qwen3_VL::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

void Qwen3_VL::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  if (token_length <= 0 || token_length >= SEQLEN || token_length > MAX_INPUT_LENGTH)
    throw std::invalid_argument("Prompt exceeds compiled prefill/sequence capacity");

  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(p_tokens, p_tokens + token_length, visited_tokens.data());
  if (!support_history) {
    assert(token_length <= MAX_INPUT_LENGTH);
  }
  empty(bm_handle, dev_buffer);
  for (int i = 0; i < token_length; i += MAX_INPUT_LENGTH) {
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_embed, in_tensors, out_tensors);
    int real_len = std::min(MAX_INPUT_LENGTH, token_length - i);
    if (real_len != MAX_INPUT_LENGTH) {
      empty(bm_handle, in_tensors[0].device_mem);
    }
    bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                          (void *)(visited_tokens.data() + i),
                          real_len * sizeof(int));
    net_launch(net_embed, in_tensors, out_tensors);
    int offset = i * HIDDEN_SIZE * sizeof(uint16_t);
    d2d(dev_buffer, out_tensors[0].device_mem, offset,
        real_len * HIDDEN_SIZE * sizeof(uint16_t));
  }
  for (auto &mem : deepstack_buffers) {
    empty(bm_handle, mem);
  }
}

void Qwen3_VL::forward_vit(ArrayFloat const &pixel_values,
                           ArrayInt const &position_ids,
                           ArrayInt const &pos_idx,
                           ArrayFloat const &pos_weight,
                           ArrayInt const &grid_thw, int vit_offset) {
  auto p_grid_thw = grid_thw.request();
  auto p_thw = static_cast<int *>(p_grid_thw.ptr);
  if (grid_thw.size() != 3) throw std::invalid_argument("grid_thw requires 3 values");
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  if (t != 1 || h <= 0 || w <= 0 || h % 2 || w % 2 || hw > MAX_PATCHES ||
      vit_offset < 0 || vit_offset + hw / 4 > token_length)
    throw std::invalid_argument("Invalid visual grid or visual-token offset");
  int num_pixels = hw * VIT_DIMS;
  assert(pixel_values.size() == num_pixels);
  assert(position_ids.size() == (hw * 2));
  assert(pos_idx.size() == (hw * 4));
  assert(pos_weight.size() == (hw * 4));
  auto p_pixel_values = pixel_values.request();
  auto p_position_ids = position_ids.request();
  auto p_pos_idx = pos_idx.request();
  auto p_pos_weight = pos_weight.request();

  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_pixel_values.ptr, num_pixels * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_position_ids.ptr,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)p_pos_idx.ptr, pos_idx.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)p_pos_weight.ptr,
                        pos_weight.size() * sizeof(float));
  if (vit_dynamic) {
    in_tensors[0].shape.dims[0] = hw;
    in_tensors[1].shape.dims[0] = hw;
    in_tensors[2].shape.dims[0] = hw;
    in_tensors[3].shape.dims[0] = hw;
  } else {
    std::vector<float> attention_mask(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    for (int i = 0; i < hw; i++) {
      auto row_begin = attention_mask.begin() + i * MAX_PATCHES;
      std::fill(row_begin, row_begin + hw, 0.0f);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[4].device_mem,
                  (void *)attention_mask.data());
  }

  net_launch(net_vit, in_tensors, out_tensors);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
  for (int i = 0; i < num_deepstack; i++) {
    bm_memcpy_d2d_byte(bm_handle, deepstack_buffers[i], dst_offset,
                       out_tensors[i + 1].device_mem, 0, vit_size);
  }
  vit_run = true;
}

int Qwen3_VL::generate(bm_device_mem_t &logits_mem) {
  const int vocab = net_lm->stages[0].output_shapes[0].dims[1];
  std::vector<float> logits(vocab);
  auto dtype = net_lm->output_dtypes[0];
  bm_status_t status;
  if (dtype == BM_FLOAT32) {
    status = bm_memcpy_d2s_partial(bm_handle, logits.data(), logits_mem, vocab * sizeof(float));
  } else if (dtype == BM_FLOAT16 || dtype == BM_BFLOAT16) {
    std::vector<uint16_t> raw(vocab);
    status = bm_memcpy_d2s_partial(bm_handle, raw.data(), logits_mem, vocab * sizeof(uint16_t));
    for (int i = 0; i < vocab; ++i)
      logits[i] = dtype == BM_FLOAT16 ? edgeforge_fp16(raw[i]) : edgeforge_bf16(raw[i]);
  } else {
    throw std::runtime_error("Unsupported lm_head logits dtype; require fp32/fp16/bf16");
  }
  if (status != BM_SUCCESS) throw std::runtime_error("Failed to read lm_head logits");
  return edgeforge_sample(logits, visited_tokens, token_length, do_sample, temperature,
                          top_p, top_k, repetition_penalty, sgen);
}

int Qwen3_VL::forward_first(ArrayInt const &position_ids) {
  if (support_history) {
    return forward_first_with_kv(position_ids);
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    if (is_dynamic) {
      attention_mask.assign(token_length * token_length, mask_value);
      for (int i = 0; i < token_length; i++) {
        for (int j = 0; j <= i; j++) {
          attention_mask[i * token_length + j] = 0;
        }
      }
    } else {
      attention_mask.assign(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, mask_value);
      for (int i = 0; i < token_length; i++) {
        for (int j = 0; j <= i; j++) {
          attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
        }
      }
    }
  }
  if (is_dynamic) {
    position_ids_pad.assign(3 * token_length, 0);
    assert((int)position_ids.size() == token_length * 3);
    std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());
  } else {
    position_ids_pad.assign(3 * MAX_INPUT_LENGTH, 0);
    int ori_length = position_ids.size() / 3;
    for (int i = 0; i < 3; i++) {
      int ori_offset = i * ori_length;
      int dst_offset = i * MAX_INPUT_LENGTH;
      std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
                position_ids_pad.begin() + dst_offset);
    }
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    if (is_dynamic) {
      d2d(in_tensors[0].device_mem, out_mem);
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                              (void *)position_ids_pad.data(),
                              token_length * 3 * sizeof(int));
        if (prefill_mask) {
          bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                                (void *)attention_mask.data(),
                                token_length * token_length * sizeof(uint16_t));
        }
      }
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
      if (prefill_mask) {
        in_tensors[2].shape.dims[2] = token_length;
        in_tensors[2].shape.dims[3] = token_length;
      }
    } else {
      in_tensors[0].device_mem = out_mem;
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                      (void *)position_ids_pad.data());
        if (prefill_mask) {
          bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                        (void *)attention_mask.data());
        }
      }
    }
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    if (vit_run && (idx < num_deepstack)) {
      init_tensors(net_add, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      in_tensors[1].device_mem = deepstack_buffers[idx];
      net_launch(net_add, in_tensors, out_tensors);
      out_mem = net_add->stages[0].output_mems[0];
    }
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[1], 0,
                       KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[2], 0,
                       KV_BYTES * token_length);
  }
  vit_run = false;

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  net_launch(net_lm, in_tensors, out_tensors);
  auto token = generate(out_tensors[0].device_mem);
  visited_tokens[token_length] = token;
  token_length++;
  history_length = token_length;
  return token;
}

int Qwen3_VL::forward_first_with_kv(ArrayInt const &position_ids) {
  assert(history_length + token_length < SEQLEN);
  assert(prefill_mask == false);
  assert((int)position_ids.size() == 3 * token_length);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> pos_ids(3 * MAX_INPUT_LENGTH, 0);

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  const int k_idx = 2; // history k/v input index (mask is baked into the net)
  int old_kvlen = (history_length > 0) ? (history_length - 1) : 0;
  int last_cur_len = 0; // cur_len of the last chunk, for lmhead offset
  for (int t = 0; t < token_length; t += MAX_INPUT_LENGTH) {
    auto old_length = history_length;
    int cur_len = std::min(MAX_INPUT_LENGTH, token_length - t);
    last_cur_len = cur_len;
    history_length += cur_len;
    // copy position ids with offset
    for (int i = 0; i < 3; i++) {
      std::copy(p_ids + i * token_length + t,
                p_ids + i * token_length + t + cur_len,
                pos_ids.data() + i * cur_len);
    }

    assert(old_length <= PREFILL_KV_LENGTH);
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
      auto &net = old_kvlen > 0 ? net_blocks_kv[idx] : net_blocks[idx];
      init_tensors(net, in_tensors, out_tensors);
      out_tensors[0].device_mem = out_mem;

      if (idx == 0) {
        // for the first block, we need to copy input from dev_buffer to
        // in_tensors[0] with offset
        bm_memcpy_d2d_byte(bm_handle, in_tensors[0].device_mem, 0, dev_buffer,
                           t * HIDDEN_SIZE * sizeof(uint16_t),
                           cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      } else {
        d2d(in_tensors[0].device_mem, out_mem, 0,
            cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      }
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)(pos_ids.data()),
                            cur_len * 3 * sizeof(int));
      in_tensors[0].shape.dims[1] = cur_len;
      in_tensors[1].shape.dims[1] = cur_len;
      // copy old kv to new kv with offset
      if (old_kvlen > 0) {
        d2d(in_tensors[k_idx].device_mem, past_key[idx], 0,
            KV_BYTES * old_kvlen);
        d2d(in_tensors[k_idx + 1].device_mem, past_value[idx], 0,
            KV_BYTES * old_kvlen);
        in_tensors[k_idx].shape.dims[1] = old_kvlen;
        in_tensors[k_idx + 1].shape.dims[1] = old_kvlen;
      }

      net_launch(net, in_tensors, out_tensors);
      if (vit_run && (idx < num_deepstack)) {
        // hidden += deepstack features of the current chunk: net_add reads
        // out_mem plus the deepstack chunk of the current chunk, and the
        // sum is copied back into out_mem.
        init_tensors(net_add, in_tensors, out_tensors);
        in_tensors[0].device_mem = out_mem;
        out_tensors[0].device_mem = out_mem;
        bm_memcpy_d2d_byte(bm_handle, in_tensors[1].device_mem, 0,
                           deepstack_buffers[idx],
                           t * HIDDEN_SIZE * sizeof(uint16_t),
                           cur_len * HIDDEN_SIZE * sizeof(uint16_t));
        net_launch(net_add, in_tensors, out_tensors);
      }
      size_t offset = old_kvlen * KV_BYTES;
      bm_memcpy_d2d_byte(bm_handle, past_key[idx], offset,
                         net->stages[0].output_mems[1], 0, KV_BYTES * cur_len);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx], offset,
                         net->stages[0].output_mems[2], 0, KV_BYTES * cur_len);
    }
    old_kvlen += cur_len;
  }
  vit_run = false;
  // forward lmhead
  // the last chunk writes its block output to dev_buffer[0..last_cur_len),
  // so the last token's hidden state is at (last_cur_len - 1) * bytes,
  // not (token_length - 1) * bytes (which is stale/garbage when chunked).
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (last_cur_len - 1) * bytes, bytes);
  net_launch(net_lm, in_tensors, out_tensors);
  int token = generate(out_tensors[0].device_mem);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int Qwen3_VL::select_decode_stage() {
  if (decode_stage_len.empty()) {
    return 0;
  }
  int stage_idx = 0;
  for (auto &len : decode_stage_len) {
    if (history_length > len) {
      break;
    }
    stage_idx++;
  }
  if (stage_idx > 0) {
    stage_idx--;
  }
  return stage_idx;
}

int Qwen3_VL::forward_next(ArrayInt const &position_ids) {
  if (history_length >= SEQLEN || token_length >= SEQLEN)
    throw std::invalid_argument("Compiled KV cache is full");
  int stage = select_decode_stage();
  int real_len = decode_stage_len.empty() ? SEQLEN : decode_stage_len[stage];
  std::vector<uint16_t> attention_mask(real_len + 1, 0);
  for (int i = history_length - 1; i < real_len; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, p_ids, attention_mask, stage);
    out_mem = net_blocks_cache[idx]->stages[stage].output_mems[0];
  }

  // forward lmhead
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = out_mem;
  net_launch(net_lm, in_tensors, out_tensors);

  token = generate(out_tensors[0].device_mem);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

PYBIND11_MODULE(edgeforge_qwen3_vl, m) {
  m.attr("EDGEFORGE_ABI") = 1;
  m.attr("UPSTREAM_REVISION") = "2866d403dae89c4164d8b96c8c615d7f67693e6d";
  pybind11::class_<Qwen3_VL>(m, "Qwen3_VL")
      .def(pybind11::init<>())
      .def("init", &Qwen3_VL::init)
      .def("configure_generation", &Qwen3_VL::configure_generation)
      .def("set_seed", &Qwen3_VL::set_seed)
      .def_readonly("NUM_LAYERS", &Qwen3_VL::NUM_LAYERS)
      .def_readonly("HIDDEN_SIZE", &Qwen3_VL::HIDDEN_SIZE)
      .def_readonly("VIT_DIMS", &Qwen3_VL::VIT_DIMS)
      .def_readonly("num_deepstack", &Qwen3_VL::num_deepstack)
      .def_readonly("lmhead_with_topk", &Qwen3_VL::lmhead_with_topk)
      .def("forward_embed", &Qwen3_VL::forward_embed)
      .def("forward_vit", &Qwen3_VL::forward_vit)
      .def("forward_first", &Qwen3_VL::forward_first)
      .def("forward_next", &Qwen3_VL::forward_next)
      .def("clear_history", &Qwen3_VL::clear_history)
      .def("deinit", &Qwen3_VL::deinit)
      .def_readonly("SEQLEN", &Qwen3_VL::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_PIXELS", &Qwen3_VL::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &Qwen3_VL::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &Qwen3_VL::MAX_INPUT_LENGTH)
      .def_readonly("PREFILL_KV_LENGTH", &Qwen3_VL::PREFILL_KV_LENGTH)
      .def_readonly("support_history", &Qwen3_VL::support_history)
      .def_readonly("history_length", &Qwen3_VL::history_length);
}

