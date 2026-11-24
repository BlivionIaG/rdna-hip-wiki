red_sync(float*, float const*):
        s_load_dwordx4 s[0:3], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        v_mbcnt_lo_u32_b32 v2, -1, 0
        v_bfi_b32 v3, v2, 0, 32
        v_xor_b32_e32 v4, 16, v2
        v_xor_b32_e32 v5, 8, v2
        v_cmp_lt_u32_e32 vcc_lo, v4, v3
        v_cndmask_b32_e32 v4, v2, v4, vcc_lo
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        s_waitcnt lgkmcnt(0)
        global_load_dword v1, v0, s[2:3]
        v_lshlrev_b32_e32 v4, 2, v4
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt vmcnt(0)
        ds_bpermute_b32 v4, v4, v1
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 4, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 2, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 1, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v2, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v2, 2, v2
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v2, v2, v1
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v2
        global_store_dword v0, v1, s[0:1]
        s_endpgm

red_nosync(float*, float const*):
        s_load_dwordx4 s[0:3], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        v_mbcnt_lo_u32_b32 v2, -1, 0
        v_bfi_b32 v3, v2, 0, 32
        v_xor_b32_e32 v4, 16, v2
        v_xor_b32_e32 v5, 8, v2
        v_cmp_lt_u32_e32 vcc_lo, v4, v3
        v_cndmask_b32_e32 v4, v2, v4, vcc_lo
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        s_waitcnt lgkmcnt(0)
        global_load_dword v1, v0, s[2:3]
        v_lshlrev_b32_e32 v4, 2, v4
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt vmcnt(0)
        ds_bpermute_b32 v4, v4, v1
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 4, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 2, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v5, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v5, 2, v5
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v4, v5, v1
        v_xor_b32_e32 v5, 1, v2
        v_cmp_lt_u32_e32 vcc_lo, v5, v3
        v_cndmask_b32_e32 v2, v2, v5, vcc_lo
        v_lshlrev_b32_e32 v2, 2, v2
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v4
        ds_bpermute_b32 v2, v2, v1
        s_waitcnt lgkmcnt(0)
        v_add_f32_e32 v1, v1, v2
        global_store_dword v0, v1, s[0:1]
        s_endpgm

vote(unsigned long long*, int const*):
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

ws(int*):
        s_load_dwordx2 s[0:1], s[4:5], 0x0
        v_lshlrev_b32_e32 v0, 2, v0
        v_mov_b32_e32 v1, 32
        s_waitcnt lgkmcnt(0)
        global_store_dword v0, v1, s[0:1]
        s_endpgm

__hip_cuid_5da1c4ac02b278ba:
        .byte   0

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
.gfx1250_revision: B0
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 16
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           red_sync(float*, float const*)
.private_segment_fixed_size: 0
.sgpr_count:     8
.sgpr_spill_count: 0
.symbol:         red_sync(float*, float const*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     6
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
.gfx1250_revision: B0
.group_segment_fixed_size: 0
.kernarg_segment_align: 8
.kernarg_segment_size: 16
.language:       OpenCL C
      - 2
      - 0
.max_flat_workgroup_size: 1024
.name:           red_nosync(float*, float const*)
.private_segment_fixed_size: 0
.sgpr_count:     8
.sgpr_spill_count: 0
.symbol:         red_nosync(float*, float const*).kd
.uniform_work_group_size: 1
.uses_dynamic_stack: false
.vgpr_count:     6
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
.gfx1250_revision: B0
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
.gfx1250_revision: B0
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
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
  - 1
  - 2

