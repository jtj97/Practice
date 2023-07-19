import torch
import torch.nn as nn

class DotAttention(nn.Module):
    def __init__(self, dim, bias = False, attn_drop = 0):
        super().__init__()
        self.norm = dim ** -0.5 # sqrt(d_k) in denominator
        
        self.proj_q = nn.Linear(dim, dim, bias = bias)
        self.proj_k = nn.Linear(dim, dim, bias = bias)
        self.proj_v = nn.Linear(dim, dim, bias = bias)
        
        self.attn_drop = nn.Dropout(p = attn_drop)
    
    def forward(self, x):
        # q,k,v (N, patch_num, D)
        q = self.proj_q(x)
        k = self.proj_q(x)
        v = self.proj_q(x)
        
        attn = q @ k.transpose(-1, -2) # (N, patch_num, D) ---> (N, patch_num, patch_num). i.e. attention matrix
        attn /= self.norm # attn/ sqrt(d_k)
        attn = attn.softmax(dim = -1) # softmax in each row
        
        res = attn @ v # res (N, patch_num, D)
        
        return res
    
class MSA(nn.Module):
    # we don't use decoder in VIT, so mask parameter is removed
    def __init__(self, dim, num_heads = 8, bias = False, attn_drop = 0, proj_drop = 0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.dim_head = dim // num_heads # dim of each head
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(DotAttention(self.dim_head, bias, attn_drop))
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p = proj_drop)
        
    def forward(self, x):
        N, N_P, D = x.shape
        
        attn_res = []
        x = x.reshape(N, N_P, self.num_heads, self.dim_head)
        for i in range(self.num_heads):
            attn_res.append(self.attention_heads[i](x[:, :, i, :]))
        y = torch.cat(attn_res, dim = 2)
        y = self.proj(y)
        y = self.proj_drop(y)
        
        return y

if __name__ == '__main__':
    import test.testMain as test
    
    dim = 768
    module = DotAttention(dim)
    test.testSequence(module) # input(32, 196, 768) --> attn: (32, 196, 196)

    module2 = MSA(dim)
    test.testSequence(module2)
    