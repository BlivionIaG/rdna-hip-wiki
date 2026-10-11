#!/usr/bin/env python3
"""Check that every mma-probe kernel contains exactly the instruction under test.

usage: python3 check_isa.py build/<arch> <arch>

Reads build/<arch>/<variant>/<family>.s (and .dis.s / .notes.txt when present),
and for every kernel mmaprobe_<op>__<mode>:
  * lists the floating-point / dot / wmma ALU mnemonics it contains,
  * PASSes only if that multiset equals the expected one for the op and arch
    (no compiler fallback, no extra conversion or FP op),
  * checks the s_setreg MODE immediate for non-desc modes (and its absence for desc),
  * records the kernel-descriptor float mode fields (.amdhsa_float_*, ieee, dx10).
Writes <dir>/isa-check.json; prints a text table (build.sh tees it to isa-check.txt).
"""
import json
import os
import re
import sys
from collections import Counter

# Mnemonics that do arithmetic on FP or packed data. Anything matching this in a
# probe kernel must be the instruction under test.
ARITH = re.compile(r"^v_(dot\d|wmma|swmmac|fma|fmac|mad|mac|mul_|add_f|sub_f|subrev_f|pk_fma|pk_mul|pk_add|"
                   r"cvt_f32_f16|cvt_f16_f32|cvt_f32_bf16|fmamk|fmaak|ldexp|min_f|max_f|med3_f)")

INTEGER = re.compile(r"_(u|i)(8|16|24|32|64)(_|$)")  # address arithmetic etc.


def is_arith(mn):
    if mn.startswith("v_dot"):
        return True
    return bool(ARITH.match(mn)) and not INTEGER.search(mn)


def any_of(*m):
    return [Counter({x: 1}) for x in m]

EXPECT = {
    # op: {arch: [acceptable Counters]}   ("*" = any arch)
    "dot2_f32_f16": {"*": any_of("v_dot2_f32_f16")},
    "dot2c_f32_f16": {"gfx1030": any_of("v_dot2c_f32_f16"), "gfx1100": any_of("v_dot2acc_f32_f16")},
    "fdot2_builtin": {"gfx1030": any_of("v_dot2c_f32_f16", "v_dot2_f32_f16"),
                      "gfx1100": any_of("v_dot2acc_f32_f16", "v_dot2_f32_f16")},
    "fdot2_builtin_clamp": {"*": any_of("v_dot2_f32_f16")},
    "dot2_f32_bf16": {"gfx1100": any_of("v_dot2_f32_bf16")},
    "fdot2_f32_bf16_builtin": {"gfx1100": any_of("v_dot2_f32_bf16")},
    "wmma_f32_16x16x16_f16": {"gfx1100": any_of("v_wmma_f32_16x16x16_f16")},
    "wmma_f32_16x16x16_bf16": {"gfx1100": any_of("v_wmma_f32_16x16x16_bf16")},
    "fma_mix_f32": {"*": any_of("v_fma_mix_f32")},
    "fma_mix_c": {"*": any_of("v_fma_mix_f32")},
    "fma_f32": {"*": any_of("v_fma_f32")},
    "fma_c": {"*": any_of("v_fma_f32", "v_fmac_f32")},
    "mul_add_c": {"*": [Counter({"v_mul_f32": 1, "v_add_f32": 1})]},
    "pk_fma_f16": {"*": any_of("v_pk_fma_f16")},
    "dot4_i32_i8": {"gfx1030": any_of("v_dot4_i32_i8")},
    "dot4_i32_i8_clamp": {"gfx1030": any_of("v_dot4_i32_i8")},
    "dot4c_i32_i8": {"gfx1030": any_of("v_dot4c_i32_i8")},
    "sdot4_builtin": {"gfx1030": any_of("v_dot4c_i32_i8", "v_dot4_i32_i8")},
    "sudot4_ss_builtin": {"gfx1100": any_of("v_dot4_i32_iu8")},
    "sudot4_ss_builtin_clamp": {"gfx1100": any_of("v_dot4_i32_iu8")},
}
# Ops whose instruction must carry an explicit modifier.
NEED_TEXT = {"dot4_i32_i8_clamp": "clamp", "fdot2_builtin_clamp": "clamp",
             "sudot4_ss_builtin_clamp": "clamp", "sudot4_ss_builtin": "neg_lo:[1,1,0]"}

KRE = re.compile(r"^(mmaprobe_(\w+?)__(\w+)):\s*(;.*)?$")
DESC_KEYS = ["float_round_mode_32", "float_round_mode_16_64", "float_denorm_mode_32",
             "float_denorm_mode_16_64", "ieee_mode", "dx10_clamp"]


def norm(mn):
    return re.sub(r"_(e32|e64|dpp|sdwa|e64_dpp)$", "", mn)


def parse_s(path):
    kernels, cur = {}, None
    descs, dcur = {}, None
    for line in open(path):
        m = KRE.match(line.strip())
        if m:
            cur = m.group(1)
            kernels[cur] = {"op": m.group(2), "mode": m.group(3), "insts": []}
            continue
        s = line.strip()
        if s.startswith(".Lfunc_end"):
            cur = None
        elif cur and s and not s.startswith((";", ".", "//")) and not s.endswith(":"):
            kernels[cur]["insts"].append(s)
        m = re.match(r"\.amdhsa_kernel\s+(\S+)", s)
        if m:
            dcur = m.group(1)
            descs[dcur] = {}
        elif s.startswith(".end_amdhsa_kernel"):
            dcur = None
        elif dcur:
            m = re.match(r"\.amdhsa_(\w+)\s+(\S+)", s)
            if m and m.group(1) in DESC_KEYS:
                descs[dcur][m.group(1)] = int(m.group(2))
    return kernels, descs


def dis_mnemonics(path):
    """Final code object disassembly: kernel -> arith mnemonics (cross-check)."""
    out, cur = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"^[0-9a-f]+ <(mmaprobe_\w+)>:", line)
        if m:
            cur = m.group(1)
            out[cur] = Counter()
            continue
        if cur:
            m = re.match(r"^\s+(v_\w+)", line)
            if m and is_arith(norm(m.group(1))):
                out[cur][norm(m.group(1))] += 1
    return out


def kd_fields(co_path):
    """Decode COMPUTE_PGM_RSRC1 float fields from every *.kd symbol in an AMDGPU
    code object (ELF64). Layout per LLVM AMDGPUUsage "Kernel Descriptor":
    compute_pgm_rsrc1 is the u32 at byte offset 48 of the 64-byte descriptor."""
    import struct
    out = {}
    if not os.path.exists(co_path):
        return out
    b = open(co_path, "rb").read()
    if b[:4] != b"\x7fELF":
        return out
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    for sec in secs:
        if sec[1] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[sec[6]]
        for j in range(sec[5] // 24):
            name_off, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", b, sec[4] + j * 24)
            name = b[strtab[4] + name_off:b.index(b"\0", strtab[4] + name_off)].decode()
            if not name.endswith(".kd") or shndx == 0 or shndx >= len(secs):
                continue
            tgt = secs[shndx]
            r1, = struct.unpack_from("<I", b, tgt[4] + (value - tgt[3]) + 48)
            out[name[:-3]] = {"float_round_mode_32": (r1 >> 12) & 3, "float_round_mode_16_64": (r1 >> 14) & 3,
                              "float_denorm_mode_32": (r1 >> 16) & 3, "float_denorm_mode_16_64": (r1 >> 18) & 3,
                              "dx10_clamp": (r1 >> 21) & 1, "ieee_mode": (r1 >> 23) & 1}
    return out


def check(dirname, arch):
    rows = []
    for var in sorted(os.listdir(dirname)):
        vdir = os.path.join(dirname, var)
        if not os.path.isdir(vdir):
            continue
        for f in sorted(os.listdir(vdir)):
            if not f.endswith(".s") or f.endswith(".dis.s"):
                continue
            fam = f[:-2]
            kernels, descs = parse_s(os.path.join(vdir, f))
            dis = dis_mnemonics(os.path.join(vdir, fam + ".dis.s"))
            kds = kd_fields(os.path.join(vdir, fam + ".co"))
            for name, k in sorted(kernels.items()):
                op, mode = k["op"], k["mode"]
                mn = [norm(i.split()[0]) for i in k["insts"]]
                arith = Counter(x for x in mn if is_arith(x))
                exp = EXPECT.get(op, {})
                accept = exp.get(arch, exp.get("*"))
                problems = []
                if accept is None:
                    problems.append("op not expected on this arch")
                elif arith not in accept:
                    problems.append("arith ops %s, expected one of %s" % (
                        dict(arith), [dict(a) for a in accept]))
                if op in NEED_TEXT:
                    hit = [i for i in k["insts"] if is_arith(norm(i.split()[0]))]
                    if not any(NEED_TEXT[op] in i for i in hit):
                        problems.append("missing modifier '%s'" % NEED_TEXT[op])
                setregs = [i for i in k["insts"] if i.startswith("s_setreg")]
                if mode == "desc":
                    if setregs:
                        problems.append("unexpected s_setreg in desc kernel")
                else:
                    want = int(mode[1:], 16)
                    ok = any(re.search(r"hwreg\(HW_REG_MODE, 0, 8\),\s*(0x[0-9a-f]+|\d+)", s) and
                             int(re.search(r",\s*(0x[0-9a-f]+|\d+)\s*$", s).group(1), 0) == want
                             for s in setregs)
                    if not ok:
                        problems.append("s_setreg MODE=0x%02x not found" % want)
                if name in dis and dis[name] != arith:
                    problems.append("code object disassembly differs: %s" % dict(dis[name]))
                d = descs.get(name, {})
                if kds:
                    if name not in kds:
                        problems.append("no .kd in code object")
                    elif kds[name] != d:
                        problems.append("code-object descriptor %s != .s directives %s" % (kds[name], d))
                rows.append({"variant": var, "family": fam, "kernel": name, "op": op, "mode": mode,
                             "arith": dict(arith), "descriptor": d,
                             "verdict": "PASS" if not problems else "FAIL", "problems": problems,
                             "instruction_text": [i for i in k["insts"] if is_arith(norm(i.split()[0]))]})
    return rows


def main():
    dirname, arch = sys.argv[1], sys.argv[2]
    rows = check(dirname, arch)
    json.dump(rows, open(os.path.join(dirname, "isa-check.json"), "w"), indent=1)
    tc = os.path.join(dirname, "toolchain.txt")
    if os.path.exists(tc):
        print(open(tc).read().rstrip())
    print()
    print("## compile flags per variant/family")
    for var in sorted(os.listdir(dirname)):
        fp = os.path.join(dirname, var, "flags.txt")
        if os.path.exists(fp):
            for line in open(fp):
                print("%-8s %s" % (var, line.rstrip()))
    print()
    # Descriptor summary per variant/family (all kernels in a file should agree).
    print("## kernel descriptor float fields (.amdhsa_* in the .s; cross-checked against")
    print("## COMPUTE_PGM_RSRC1 decoded from the .kd symbols of the code object)")
    seen = {}
    for r in rows:
        key = (r["variant"], r["family"])
        seen.setdefault(key, set()).add(tuple(sorted(r["descriptor"].items())))
    for (v, f), ds in sorted(seen.items()):
        for d in ds:
            print("%-8s %-13s %s" % (v, f, " ".join("%s=%s" % kv for kv in d)))
    print()
    print("## instruction under test, per op (all modes collapsed)")
    agg = {}
    for r in rows:
        key = (r["variant"], r["op"])
        a = agg.setdefault(key, {"n": 0, "fail": [], "ops": set(), "text": set()})
        a["n"] += 1
        a["ops"].add(json.dumps(r["arith"], sort_keys=True))
        a["text"].update(r["instruction_text"][:1])
        if r["verdict"] != "PASS":
            a["fail"].append("%s: %s" % (r["mode"], "; ".join(r["problems"])))
    for (v, op), a in sorted(agg.items()):
        print("%-4s %-8s %-24s kernels=%-2d %s" % ("PASS" if not a["fail"] else "FAIL", v, op, a["n"],
                                                  " | ".join(sorted(a["ops"]))))
        print("      e.g. %s" % (sorted(a["text"])[0] if a["text"] else "-"))
        for p in a["fail"][:5]:
            print("      FAIL %s" % p)
    nfail = sum(r["verdict"] != "PASS" for r in rows)
    print()
    print("TOTAL kernels=%d fail=%d -> %s" % (len(rows), nfail, "ISA CHECK PASS" if nfail == 0 else "ISA CHECK FAIL"))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
