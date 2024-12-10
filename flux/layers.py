import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from comfy.ldm.flux.layers import SingleStreamBlock as OriginalSingleStreamBlock, DoubleStreamBlock as OriginalDoubleStreamBlock
from comfy.ldm.modules.attention import optimized_attention

from ..utils.rave_rope_attention import rave_rope_attention
from ..utils.flow_attention import flow_attention


def apply_rope_single(xq: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, skip_rope: bool= False, k_pe=None, mask=None) -> Tensor:
    if not skip_rope:
        q_pe = pe
        q = apply_rope_single(q, q_pe)

    if k_pe is None:
        k_pe = pe   
    k = apply_rope_single(k, k_pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)

    return x


def ref_attention(q, k, v, pe, ref_config, ref_type, idx, txt_shape=256):
    ref_pes = ref_config['ref_pes']
    k2 = torch.cat([k[:1], k[1:]], dim=2)
    v2 = torch.cat([v[:1], v[1:]], dim=2)
    attn_a = attention(q[:1], k2, v2, pe=pe[:1], k_pe=torch.cat([pe[:1], ref_pes[0]], dim=2))
    attn = attention(q[1:], k[1:], v[1:], pe=pe[:1])
    attn = torch.cat([attn_a, attn])
    max_val = 1.0
    attn2 = attention(q[:1], k[1:], v[1:], pe=pe[:1], skip_rope=False, k_pe=pe[:1])
    img_attn1 = attn[:1, 256 :]
    img_attn1 = attn[:1, txt_shape :]
    img_attn2 = attn2[:1, txt_shape :]
    strength = min(max_val, ref_config[f'strengths'][ref_config['step']])
    attn[:1, txt_shape:] = img_attn1*(1-strength) + img_attn2*strength
    attn[1:, :256] = attn[:1, :256]
    return attn


class DoubleStreamBlock(OriginalDoubleStreamBlock):
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, ref_config, timestep, transformer_options={}):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        post_q_fn = transformer_options.get('patches_replace', {}).get(f'double', {}).get(('post_q', self.idx), None) 
        if post_q_fn is not None:
            q = post_q_fn(q, transformer_options)

        # Mask Patch
        mask_fn = transformer_options.get('patches_replace', {}).get(f'double', {}).get(('mask_fn', self.idx), None) 
        mask = None
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, 256)

        rfedit = transformer_options.get('rfedit', {})
        if rfedit.get('process', None) is not None and rfedit['double_layers'][str(self.idx)]:
            pred = rfedit['pred']
            step = rfedit['step']
            if rfedit['process'] == 'forward':
                rfedit['bank'][step][pred][self.idx] = v.cpu()
            elif rfedit['process'] == 'reverse':
                v = rfedit['bank'][step][pred][self.idx].to(v.device)

        rave_options = transformer_options.get('RAVE', None)
        if ref_config is not None and ref_config['strengths'][ref_config['step']] > 0 and self.idx <= 20:
            attn = ref_attention(q, k, v, pe, ref_config, 'double', self.idx)
        elif rave_options:
            attn = rave_rope_attention(img_q, img_k, img_v, txt_q, txt_k, txt_v, pe, transformer_options, self.num_heads, 256)
        else:
            attn = attention(q, k, v, pe=pe, mask=mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        txt_attn = txt_attn[0:1].repeat(img_attn.shape[0], 1, 1)

        # if self.idx % 8 == 0:
        #     img_attn = flow_attention(img_attn, self.num_heads, self.hidden_size // self.num_heads, transformer_options)

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(OriginalSingleStreamBlock):
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, ref_config, timestep, transformer_options={}) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        post_q_fn = transformer_options.get('patches_replace', {}).get(f'single', {}).get(('post_q', self.idx), None) 
        if post_q_fn is not None:
            q = post_q_fn(q, transformer_options)

        mask_fn = transformer_options.get('patches_replace', {}).get(f'single', {}).get(('mask_fn', self.idx), None) 
        mask = None
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, 256)

        rfedit = transformer_options.get('rfedit', {})
        if rfedit.get('process', None) is not None and rfedit['single_layers'][str(self.idx)]:
            pred = rfedit['pred']
            step = rfedit['step']
            if rfedit['process'] == 'forward':
                rfedit['bank'][step][pred][self.idx] = v.cpu()
            elif rfedit['process'] == 'reverse':
                v = rfedit['bank'][step][pred][self.idx].to(v.device)

        rave_options = transformer_options.get('RAVE', None)
        if ref_config is not None and ref_config['single_strength'] > 0 and self.idx < 10:
            attn = ref_attention(q, k, v, pe, ref_config, 'single', self.idx)
        elif rave_options is not None:
            txt_q, img_q = q[:,:,:256], q[:,:,256:]
            txt_k, img_k = k[:,:,:256], k[:,:,256:]
            txt_v, img_v = v[:,:,:256], v[:,:,256:]
            attn = rave_rope_attention(img_q, img_k, img_v, txt_q, txt_k, txt_v, pe, transformer_options, self.num_heads, 256)
        else:
            attn = attention(q, k, v, pe=pe, mask=mask)
        _, img_attn = attn[:, :256], attn[:, 256:]
        
        # if self.idx % 8 == 0:
        #     img_attn = flow_attention(img_attn, self.num_heads, self.hidden_dim//self.num_heads, transformer_options)
        
        # Ensure img_attn matches expected size before assignment
        if img_attn.shape[-2] > attn[:, 256:].shape[-2]:
            img_attn = img_attn[..., :attn[:, 256:].shape[-2], :]
        
        # Assign truncated img_attn
        attn[:, 256:] = img_attn

        # Match sequence lengths for mlp concatenation
        mlp_act = self.mlp_act(mlp)
        min_seq_len = min(attn.size(-2), mlp_act.size(-2))
        attn = attn[..., :min_seq_len, :]
        mlp_act = mlp_act[..., :min_seq_len, :]
        
        # Concatenate and compute output
        output = self.linear2(torch.cat((attn, mlp_act), 2))
        return x + mod.gate * output


def inject_blocks(diffusion_model):
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = DoubleStreamBlock
        print('double_block', i, "modified")
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = SingleStreamBlock
        print('single_block', i, "modified")
        block.idx = i

    return diffusion_model