"""BF16 depth predictor shared by both arms of the synthetic benchmark."""
import types


def patch_predict_depth_bf16(model):
    """Replace model._predict_depth with a bf16-friendly version.

    Original (gct_base.py:188-208) forces fp32 via `[t.float() for t in ...]`,
    `images.float()`, and `with autocast(enabled=False):`.  This version skips
    all three so depth_head runs under the active bf16 autocast context.
    """
    def _predict_depth_bf16(self, aggregated_tokens_list, images, patch_start_idx, gather_outputs=True):
        if self.depth_head is None:
            return {}
        depth, depth_conf = self.depth_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        return {"depth": depth, "depth_conf": depth_conf}

    model._predict_depth = types.MethodType(_predict_depth_bf16, model)
