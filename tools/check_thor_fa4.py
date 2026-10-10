"""Short, separate-process paged-FA4 correctness check; no timing."""
import argparse
from pathlib import Path

from lingbot_map.optimizations.thor import ThorOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    ThorOptions(fa4_query_staging=True, paged_kv_affine=args.affine).activate()
    from lingbot_map.optimizations.thor.fa4_runtime import prepare_fa4_overlay
    prepare_fa4_overlay()
    import torch
    from flash_attn.cute.interface import _flash_attn_fwd
    torch.manual_seed(42)
    outputs = {}
    cases = [(tokens, page, "interleaved") for tokens, page in
             ((783, 777), (978, 972), (1005, 999), (1042, 1036))]
    cases.append((1005, 999, "contiguous"))
    for tokens, page, layout in cases:
        q = torch.randn(tokens, 16, 64, device="cuda", dtype=torch.bfloat16)
        cache = torch.randn(5, 2, page, 16, 64, device="cuda", dtype=torch.bfloat16)
        k, v = cache[:, 0], cache[:, 1]
        if layout == "contiguous":
            k, v = k.contiguous(), v.contiguous()
        table = torch.tensor([[3, 1, 4, 0, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([page * 3 + 37], device="cuda", dtype=torch.int32)
        cu_q = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q)
        def run():
            return _flash_attn_fwd(q, k, v,
                cu_seqlens_q=cu_q, seqused_k=lengths, max_seqlen_q=tokens,
                page_table=table, causal=False, num_splits=1, out=out)[0]
        with torch.no_grad():
            run()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            # Replay the same graph across tile tails and physical page boundaries.
            for length in (1, 127, 128, 129, page - 1, page, page + 1, page * 3 + 37, page * 5):
                lengths.fill_(length)
                expected = run().clone()
                graph.replay()
                torch.cuda.synchronize()
                key = f"q{tokens}_{layout}_kv{length}"
                if not torch.isfinite(out).all() or not torch.equal(expected, out):
                    raise RuntimeError(f"FA4 captured replay differs from eager: {key}")
                outputs[key] = out.cpu()
        del graph
    if args.affine:
        for tokens in (1004, 1006):
            page = tokens - 6
            q = torch.empty(tokens, 16, 64, device="cuda", dtype=torch.bfloat16)
            cache = torch.empty(2, 2, page, 16, 64, device="cuda", dtype=torch.bfloat16)
            try:
                _flash_attn_fwd(
                    q, cache[:, 0], cache[:, 1],
                    cu_seqlens_q=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
                    seqused_k=torch.tensor([page], device="cuda", dtype=torch.int32),
                    max_seqlen_q=tokens,
                    page_table=torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
                    causal=False, num_splits=1,
                )
            except RuntimeError as exc:
                if "contract mismatch" not in str(exc):
                    raise
            else:
                raise RuntimeError(f"Unaudited affine shape was accepted: Q={tokens}")
    if args.reference:
        reference = torch.load(args.reference, weights_only=True)
        if outputs.keys() != reference.keys():
            raise RuntimeError("Reference and candidate FA4 case sets differ")
        for key in outputs:
            if not torch.equal(outputs[key], reference[key]):
                raise RuntimeError(f"FA4 output differs for Q={key}")
        print(f"PASS: {len(outputs)} captured FA4 cases are bitwise equal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output)


if __name__ == "__main__":
    main()
