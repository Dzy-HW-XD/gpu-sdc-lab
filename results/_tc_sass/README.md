# Tensor-Core BF16 GEMM SASS (target kernel for injection)

`target_kernel.sass` = the cuBLASLt kernel used by the TC experiments:
`ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_nn` (sm_80), extracted from `libcublasLt.so.13.1.0.3` with

```
cuobjdump -sass -arch sm_80 libcublasLt.so.13.1.0.3 > _lt_sm80.sass
# then the block whose header is 'Function : ...s16816gemm_bf16_256x128_ldg8_stages_32x3_nn...'
```

Read: `/*pc*/ OPCODE ...`; `pc` matches `fault.pc_offset` in
`observations.jsonl`. Injections are `HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc`
(Rd = FP32 accumulator quad base; Rc = C; Rd==Rc = accumulate).
