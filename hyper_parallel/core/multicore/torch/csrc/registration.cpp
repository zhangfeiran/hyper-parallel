/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 *
 * PyTorch out-of-tree operator registration for MoE-FFN operators.
 * Registers into the 'hyper_parallel' namespace — does NOT modify aten:: or op-plugin.
 *
 * The packaged adapter is loaded lazily through torch.ops.load_library().
 * Static initializers register the operators, which are then called as:
 *   torch.ops.hyper_parallel.mega_moe(...)
 *   torch.ops.hyper_parallel.mega_moe_grad(...)
 */
#include <torch/library.h>

namespace {
void register_dsa_probes(torch::Library& m) {
  m.def("dsa_fused_forward_version() -> int", []() -> int64_t { return 2; });
  m.def("dsa_fused_forward_out(Tensor index_query, Tensor index_key, Tensor query, Tensor compressed, "
        "Tensor query_rope, Tensor key_rope, Tensor weights, Tensor lengths, Tensor config, "
        "Tensor(a!) trace, Tensor(b!) retained, float scale, Tensor(c!) indices, Tensor(d!) values, "
        "Tensor(e!) attention, Tensor(f!) maximum, Tensor(g!) sum) "
        "-> (Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(a!), Tensor(b!))");
  m.def("dsa_fused_cp_forward_out(Tensor index_query, Tensor(h!) index_key, Tensor query, Tensor(i!) compressed, "
        "Tensor query_rope, Tensor(j!) key_rope, Tensor weights, Tensor lengths, Tensor config, "
        "Tensor(a!) trace, Tensor(b!) retained, float scale, Tensor(c!) indices, Tensor(d!) values, "
        "Tensor(e!) attention, Tensor(f!) maximum, Tensor(g!) sum, Tensor(k!) arena, Tensor metadata, "
        "Tensor requests, Tensor(l!) transport_trace) "
        "-> (Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(a!), Tensor(b!))");
  m.def("dsa_mixed_grad_version() -> int", []() -> int64_t { return 1; });
  m.def("dsa_mixed_grad_out(Tensor query, Tensor key, Tensor value, Tensor indices, Tensor grad_out, "
        "Tensor out, Tensor maximum, Tensor sum, Tensor actual_query, Tensor actual_kv, Tensor query_rope, "
        "Tensor key_rope, Tensor config, Tensor(a!) trace, Tensor(b!) retained, float scale, int phase, "
        "Tensor(c!) grad_query, Tensor(d!) grad_key, Tensor(e!) grad_value, Tensor(f!) grad_query_rope, "
        "Tensor(g!) grad_key_rope) "
        "-> (Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(a!), Tensor(b!))");
  m.def("dsa_mixed_indexer_version() -> int", []() -> int64_t { return 2; });
  m.def("dsa_mixed_indexer_out(Tensor query, Tensor key, Tensor weights, Tensor actual_query, "
        "Tensor actual_kv, Tensor config, Tensor(a!) trace, Tensor(b!) retained, int merge_phase, "
        "Tensor(c!) indices, Tensor(d!) values) -> (Tensor(c!), Tensor(d!), Tensor(a!), Tensor(b!))");
  m.def("dsa_mixed_tile_version() -> int", []() -> int64_t { return 1; });
  m.def("dsa_mixed_tile_out(Tensor query, Tensor key, Tensor value, Tensor indices, "
        "Tensor actual_query, Tensor actual_kv, Tensor query_rope, Tensor key_rope, Tensor config, "
        "Tensor(a!) trace, float scale, Tensor(b!) out, Tensor(c!) maximum, Tensor(d!) sum) "
        "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
}
}  // namespace

TORCH_LIBRARY(hyper_parallel, m) {
  register_dsa_probes(m);
  m.def("moe_token_permute_out(Tensor tokens, Tensor indices, Tensor(a!) output, Tensor(b!) mapping) "
        "-> (Tensor(a!), Tensor(b!))");
  m.def(
    "mega_moe_unpermute_grad_out(Tensor permuted_tokens, Tensor grad_output, "
    "Tensor sorted_indices, Tensor probs, Tensor(a!) grad_permuted, Tensor(b!) grad_probs) "
    "-> (Tensor(a!), Tensor(b!))");
  m.def("mega_moe_transport_version() -> int", []() -> int64_t { return 1; });
  // -------------------------------------------------------------------------
  // mega_moe: MoE-FFN forward operator (wraps aclnnHyperMegaMoe)
  //
  // Output buffers written in-place:
  //   dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target.
  // All buffers must be pre-allocated by the caller.
  // -------------------------------------------------------------------------
  m.def(
    "mega_moe("
    "  Tensor(a!) dispatch_target,"
    "  Tensor dispatch_target_off,"
    "  Tensor dispatch_src,"
    "  Tensor dispatch_src_off,"
    "  Tensor dispatch_size,"
    "  Tensor up_proj_weight,"
    "  Tensor up_proj_glist,"
    "  Tensor(b!) up_proj_y,"
    "  Tensor(c!) swiglu_out,"
    "  Tensor down_proj_weight,"
    "  Tensor down_proj_glist,"
    "  Tensor(d!) down_proj_y,"
    "  Tensor(e!) combine_target,"
    "  Tensor combine_target_off,"
    "  Tensor combine_src_off,"
    "  Tensor combine_size,"
    "  Tensor gmm_workspace,"
    "  Tensor up_proj_tiling,"
    "  Tensor swiglu_tiling,"
    "  Tensor down_proj_tiling,"
    "  Tensor runtime_config,"
    "  Tensor all_event_counters,"
    "  Tensor profile_buffer,"
    "  int rank_id,"
    "  int ep,"
    "  int expert_num,"
    "  int hidden_size,"
    "  int seq_size"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))");

  // -------------------------------------------------------------------------
  // mega_moe_grad: MoE-FFN backward operator (wraps aclnnHyperMegaMoeGrad)
  //
  // Output buffers written in-place:
  //   dispatch_target, hidden_dw, act_grad_y, grad_gate,
  //   gate_dx, grad_x, permute_out, gate_dw.
  // All buffers must be pre-allocated by the caller.
  // -------------------------------------------------------------------------
  m.def(
    "mega_moe_grad("
    "  Tensor(a!) dispatch_target,"
    "  Tensor dispatch_target_off,"
    "  Tensor dy,"
    "  Tensor dispatch_src_off,"
    "  Tensor dispatch_size,"
    "  Tensor hidden,"
    "  Tensor(b!) hidden_dw,"
    "  Tensor w2,"
    "  Tensor(c!) act_grad_y,"
    "  Tensor gate,"
    "  Tensor(d!) grad_gate,"
    "  Tensor w1,"
    "  Tensor(a!) gate_dx,"
    "  Tensor(f!) grad_x,"
    "  Tensor combine_target_off,"
    "  Tensor combine_src_off,"
    "  Tensor combine_size,"
    "  Tensor(g!) permute_out,"
    "  Tensor(h!) gate_dw,"
    "  Tensor group_list,"
    "  Tensor act_grad_tiling,"
    "  Tensor gate_grad_tiling,"
    "  Tensor w1_grad_tiling,"
    "  Tensor w2_grad_tiling,"
    "  Tensor swiglu_grad_tiling,"
    "  Tensor gmm_workspace,"
    "  Tensor swiglu_grad_workspace,"
    "  Tensor runtime_config,"
    "  Tensor all_event_counters,"
    "  Tensor profile_buffer,"
    "  int rank_id,"
    "  int ep,"
    "  int expert_num,"
    "  int hidden_size,"
    "  int seq_size"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!), Tensor(f!), Tensor(g!), Tensor(h!))");
}
