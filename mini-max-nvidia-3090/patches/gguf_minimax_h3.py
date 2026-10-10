"""Teach the ComfyUI-GGUF loaders about unsloth/MiniMax-H3-GGUF.

Those files carry no `general.architecture` and already use ComfyUI's state-dict key names
(`blocks.N...` for the DiT, `model.layers.N...` for the Qwen3-VL text encoder), so the stock
loaders reject them ("Unknown model architecture" / "incompatible with llama.cpp").
This tags them by tensor signature and loads them as-is.

Usage: python gguf_minimax_h3.py <loader.py> [<loader.py> ...]   (idempotent)
"""
import sys

MARK = "# [minimax-h3 gguf patch]"

DETECT = '''
    {mark}
    if arch_str is None:
        _keys = set(k for k, _ in tensors)
        if "adaln_t_table" in _keys and "audio_patch_proj.weight" in _keys:
            arch_str = "minimax_h3"
        elif is_text_model and "model.layers.0.mlp.gate_proj.weight" in _keys:
            arch_str = "qwen3vl_native"
'''.format(mark=MARK)

CLIP = '''    elif arch == "qwen3vl_native":  {mark}
        # Keys are already ComfyUI-native; only keep the 151k-token embedding out of GGML ops (see T5 note above).
        temb_key = "model.embed_tokens.weight"
        if temb_key in sd and hasattr(sd[temb_key], "tensor_type"):
            logging.warning(f"Dequantizing {{temb_key}} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        # GGUF tops out at 4 dims and the file has no comfy.gguf.orig_shape metadata: the vision Conv3d
        # weight [1152, 3, 2, 16, 16] was stored with its first two dims merged ([3456, 2, 16, 16]).
        pe_key = "visual.patch_embed.proj.weight"
        if pe_key in sd and sd[pe_key].ndim == 4:
            t = dequantize_tensor(sd[pe_key], dtype=torch.float16)
            sd[pe_key] = t.reshape(t.shape[0] // 3, 3, *t.shape[1:])
'''.format(mark=MARK)


def patch(path):
    s = open(path).read()
    if MARK in s:
        print(f"{path}: already patched")
        return
    for old, new in [
        ('IMG_ARCH_LIST = {', 'IMG_ARCH_LIST = {"minimax_h3", '),
        ('TXT_ARCH_LIST = {', 'TXT_ARCH_LIST = {"qwen3vl_native", '),
    ]:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    anchor = '    type_str = get_field(reader, "general.type", str)\n'
    assert s.count(anchor) == 1, (path, anchor)
    s = s.replace(anchor, anchor + DETECT)
    # gguf_clip_loader: add a branch before its final `else: pass`.
    i = s.index("def gguf_clip_loader")
    tail = "    else:\n        pass\n    return sd"
    j = s.index(tail, i)
    s = s[:j] + CLIP + s[j:]
    open(path, "w").write(s)
    print(f"{path}: patched")


for p in sys.argv[1:]:
    patch(p)
