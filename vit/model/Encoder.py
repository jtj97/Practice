import torch.nn as nn
import FFN
import MSA


class Encoder(nn.Module):
    def __init__(self, dim = 768, ffn_ratio = 4, num_heads = 8, qkv_bias = False, attn_drop = 0, drop = 0):
        super().__init__()
        self.subLayer1 = nn.Sequential(
            nn.LayerNorm(dim),
            MSA.MSA(dim, num_heads, qkv_bias, attn_drop, drop)
        )
        self.subLayer2 = nn.Sequential(
            nn.LayerNorm(dim),
            FFN.FFN(dim, dim * ffn_ratio, drop)
        )
        
    def forward(self, x):
        residual = x
        x = residual + self.subLayer1(x)
        
        residual = x
        x = residual + self.subLayer2(x)
        
        return x
    
if __name__ == '__main__':
    import test.testMain as test
    module = Encoder()
    # except output = (32,196,768)
    test.testSequence(module)
