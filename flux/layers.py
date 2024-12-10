import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from comfy.ldm.flux.layers import SingleStreamBlock as OriginalSingleStreamBlock, DoubleStreamBlock as OriginalDoubleStreamBlock
from comfy.ldm.modules.attention import optimized_attention


# Use optimized_attention as the attention function
attention = optimized_attention

def attention(q, k, v, heads=None, mask=None):
    """
    Wrapper for optimized attention with required heads parameter
    """
    if heads is None:
        raise ValueError("heads parameter is required")
    
    # Get minimum sequence length
    min_seq_len = min(q.shape[1], k.shape[1], v.shape[1])
    
    # Truncate to match minimum sequence length
    q = q[:, :min_seq_len, :]
    k = k[:, :min_seq_len, :]
    v = v[:, :min_seq_len, :]
    
    return optimized_attention(q, k, v, heads=heads, skip_reshape=True, mask=mask)


class DoubleStreamBlock(OriginalDoubleStreamBlock):
    def forward(self, img, txt, vec, pe, ref_config=None, timestep=None, transformer_options={}):
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

        def safe_concat(txt_tensor, img_tensor, dim):
            min_seq_len = min(txt_tensor.shape[2], img_tensor.shape[2])
            txt_tensor = txt_tensor[..., :min_seq_len, :]
            img_tensor = img_tensor[..., :min_seq_len, :]
            return torch.cat((txt_tensor, img_tensor), dim=dim)

        # Use safe concatenation
        q = safe_concat(txt_q, img_q, dim=2)
        k = safe_concat(txt_k, img_k, dim=2)
        v = safe_concat(txt_v, img_v, dim=2)

        post_q_fn = transformer_options.get('patches_replace', {}).get(f'double', {}).get(('post_q', self.idx), None) 
        if post_q_fn is not None:
            q = post_q_fn(q, transformer_options)

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
            attn = ref_attention(q, k, v, pe, ref_config, 'double', self.idx, self.num_heads)
        elif rave_options is not None:
            attn = rave_rope_attention(img_q, img_k, img_v, txt_q, txt_k, txt_v, pe, transformer_options, self.num_heads, 256)
        else:
            # Remove pe parameter and add mask as a keyword argument if it exists
            attn = attention(q, k, v, heads=self.num_heads, mask=mask) if mask is not None else attention(q, k, v, heads=self.num_heads)

        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        txt_attn = txt_attn[0:1].repeat(img_attn.shape[0], 1, 1)

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(OriginalSingleStreamBlock):
    def forward(self, x, vec, pe, ref_config=None, timestep=None, transformer_options={}):
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        def safe_concat(txt_tensor, img_tensor, dim):
            if len(txt_tensor.shape) == 4:  # Handle 4D tensors
                min_seq_len = min(txt_tensor.shape[2], img_tensor.shape[2])
                txt_tensor = txt_tensor[..., :min_seq_len, :]
                img_tensor = img_tensor[..., :min_seq_len, :]
            else:  # Handle 3D tensors
                min_seq_len = min(txt_tensor.shape[1], img_tensor.shape[1])
                txt_tensor = txt_tensor[:, :min_seq_len, :]
                img_tensor = img_tensor[:, :min_seq_len, :]
            return torch.cat((txt_tensor, img_tensor), dim=dim)

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
            attn = ref_attention(q, k, v, pe, ref_config, 'single', self.idx, self.num_heads)
        elif rave_options is not None:
            txt_split = 256
            txt_q, img_q = q[:,:,:txt_split], q[:,:,txt_split:]
            txt_k, img_k = k[:,:,:txt_split], k[:,:,txt_split:]
            txt_v, img_v = v[:,:,:txt_split], v[:,:,txt_split:]
            
            # Ensure consistent lengths before RAVE attention
            img_q = img_q[..., :min(img_q.shape[-2], txt_q.shape[-2]), :]
            img_k = img_k[..., :min(img_k.shape[-2], txt_k.shape[-2]), :]
            img_v = img_v[..., :min(img_v.shape[-2], txt_v.shape[-2]), :]
            
            attn = rave_rope_attention(img_q, img_k, img_v, txt_q, txt_k, txt_v, pe, transformer_options, self.num_heads, txt_split)
        else:
            # Remove pe parameter and add mask as a keyword argument if it exists
            attn = attention(q, k, v, heads=self.num_heads, mask=mask) if mask is not None else attention(q, k, v, heads=self.num_heads)
            
        _, img_attn = attn[:, :256], attn[:, 256:]
        attn[:, 256:] = img_attn

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


def inject_blocks(diffusion_model):
    """Inject the custom block implementations into the diffusion model."""
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = DoubleStreamBlock
        print('double_block', i, "modified")
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = SingleStreamBlock
        print('single_block', i, "modified")
        block.idx = i

    return diffusion_model