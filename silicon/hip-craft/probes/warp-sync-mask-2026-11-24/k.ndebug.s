red_sync(float*, float const*):                       ; @red_sync(float*, float const*)
        s_load_dwordx4 s[0:3], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        s_waitcnt lgkmcnt(0)
        global_load_dword v1, v0, s[2:3]
        s_mov_b32 s2, 0x76543210
        s_waitcnt vmcnt(0)
        v_permlanex16_b32 v2, v1, s2, 0xfedcba98
        v_add_f32_e32 v1, v1, v2
        v_add_f32_dpp v1, v1, v1 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
        global_store_dword v0, v1, s[0:1]
        s_endpgm
red_nosync(float*, float const*):                    ; @red_nosync(float*, float const*)
        s_load_dwordx4 s[0:3], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        s_waitcnt lgkmcnt(0)
        global_load_dword v1, v0, s[2:3]
        s_mov_b32 s2, 0x76543210
        s_waitcnt vmcnt(0)
        v_permlanex16_b32 v2, v1, s2, 0xfedcba98
        v_add_f32_e32 v1, v1, v2
        v_add_f32_dpp v1, v1, v1 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
        v_add_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
        global_store_dword v0, v1, s[0:1]
        s_endpgm
vote(unsigned long long*, int const*):                           ; @vote(unsigned long long*, int const*)
        s_load_dwordx4 s[0:3], s[4:5], 0x0
        v_lshlrev_b32_e32 v1, 2, v0
        v_mov_b32_e32 v2, exec_lo
        v_lshlrev_b32_e32 v4, 4, v0
        s_waitcnt lgkmcnt(0)
        global_load_dword v1, v1, s[2:3]
        s_waitcnt vmcnt(0)
        v_cmp_ne_u32_e32 vcc_lo, 0, v1
        v_mov_b32_e32 v1, 0
        v_mov_b32_e32 v0, vcc_lo
        v_mov_b32_e32 v3, v1
        global_store_dwordx4 v4, v[0:3], s[0:1]
        s_endpgm
ws(int*):                                ; @ws(int*)
        s_load_dwordx2 s[0:1], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        v_mov_b32_e32 v1, 32
        s_waitcnt lgkmcnt(0)
        global_store_dword v0, v1, s[0:1]
        s_endpgm
__hip_cuid_5cd955959cb0a5e5:
        .byte   0                               ; 0x0

---
  - .args:
      - .address_space:  global
.name:           o.coerce
.offset:         0
.size:           8
.value_kind:     global_buffer
      - .address_space:  global
.name:           i.coerce
.offset:         8
.size:           8
.value_kind:     global_buffer
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 16
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           red_sync(float*, float const*)
.private_segment_fixed_size: 0
.sgpr_count:     6
.sgpr_spill_count: 0
.symbol:         red_sync(float*, float const*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     3
.vgpr_spill_count: 0
.wavefront_size: 32
.workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
.name:           o.coerce
.offset:         0
.size:           8
.value_kind:     global_buffer
      - .address_space:  global
.name:           i.coerce
.offset:         8
.size:           8
.value_kind:     global_buffer
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 16
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           red_nosync(float*, float const*)
.private_segment_fixed_size: 0
.sgpr_count:     6
.sgpr_spill_count: 0
.symbol:         red_nosync(float*, float const*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     3
.vgpr_spill_count: 0
.wavefront_size: 32
.workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
.name:           o.coerce
.offset:         0
.size:           8
.value_kind:     global_buffer
      - .address_space:  global
.name:           p.coerce
.offset:         8
.size:           8
.value_kind:     global_buffer
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 16
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           vote(unsigned long long*, int const*)
.private_segment_fixed_size: 0
.sgpr_count:     8
.sgpr_spill_count: 0
.symbol:         vote(unsigned long long*, int const*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     5
.vgpr_spill_count: 0
.wavefront_size: 32
.workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
.name:           o.coerce
.offset:         0
.size:           8
.value_kind:     global_buffer
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 8
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           ws(int*)
.private_segment_fixed_size: 0
.sgpr_count:     6
.sgpr_spill_count: 0
.symbol:         ws(int*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     2
.vgpr_spill_count: 0
.wavefront_size: 32
.workgroup_processor_mode: 1
amdhsa.target:   amdgpu10.30-amd-amdhsa-unknown-gfx1030
  - 1
  - 2

