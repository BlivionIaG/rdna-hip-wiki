import re
t=open('rpass.txt').read()
s=open('q5k_probe-hip-amdgcn-amd-amdhsa-gfx1030.s').read()
for m in re.finditer(r"Function Name: (\S+).*?SGPRs: (\d+).*?VGPRs: (\d+).*?ScratchSize \[bytes/lane\]: (\d+).*?Occupancy \[waves/SIMD\]: (\d+).*?LDS Size \[bytes/block\]: (\d+)",t,re.S):
    n=m.group(1); a=re.search(r"ILi(\d)ELb(\d)ELi(\d)ELi(\d)ELi(\d)E",n).groups()
    body=s[s.index(n+":"):]; body=body[:body.index(".Lfunc_end")]
    c=lambda p: len(re.findall(p,body))
    v=int(m.group(3)); g=-(-v//16)*16; w=min(16,1024//g)
    print("| %s | %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %d | %d | %d |"%(a[0],"dense" if a[1]=="1" else "stock",a[2],a[3],a[4],v,g,w,m.group(5),m.group(4),m.group(6),c(r"v_dot4c?_i32_i8"),c(r"v_mul_i32_i24"),c(r"v_cvt_f32_i32")))
