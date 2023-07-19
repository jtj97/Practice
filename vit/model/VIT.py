import torch
import torch.nn as nn
import PatchEmbedding
import Encoder

class VIT(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, channels = 3, num_classes = 1000, embed_dim = 768, num_encoder = 12, ffn_ratio = 4, num_heads = 8, qkv_bias = True, drop_rate = 0, attn_drop_rate = 0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.classification_token_num = 1
        self.classification_token = nn.Parameter(torch.zeros(1, self.classification_token_num, embed_dim))
        self.pos_embed = self.position_embedding((img_size // patch_size) ** 2 + 1, embed_dim)
        
        self.patch_embedding = PatchEmbedding.PatchEmbedding(img_size, patch_size, channels, embed_dim)
        
        blocks = []
        for _ in range(num_encoder):
            blocks.append(Encoder.Encoder(embed_dim, ffn_ratio, num_heads, qkv_bias, attn_drop_rate, drop_rate))
        
        self.encoders = nn.Sequential(*blocks)
        
        self.classification_head = nn.Linear(self.embed_dim, self.num_classes)
    
    def position_embedding(self, seq_len, embed_dim):
        pos_embed = []
        for pos in range(seq_len):
            if pos == 0:
                pos_embed.append([0] * embed_dim)
                continue
            tmp = []
            for i in range(embed_dim):
                tmp.append(pos / (10000 ** (2 * i / embed_dim)))
            pos_embed.append(tmp)
                
        pos_embed = torch.Tensor(pos_embed)
        
        pos_embed[1:, 0::2] = torch.sin(pos_embed[1:, 0::2])
        pos_embed[1:, 1::2] = torch.cos(pos_embed[1:, 1::2])
        
        return pos_embed
    
    def forward(self, x):
        patch_embed = self.patch_embedding(x)
        cls_token = self.classification_token.expand(x.shape[0], -1, -1)
        embed = torch.cat((cls_token, patch_embed), dim = 1) # (N, patch_num, embed_dim) -> (N, patch_num + 1, embed_dim)
        embed += self.pos_embed
        
        embed = self.encoders(embed) # (N, patch_num + 1, embed_dim)
        res = embed[:, 0] # choose the first token
        res = self.classification_head(embed[:, 0])
        
        return res
    
if __name__ == '__main__':
    import test.testMain as test
    model = VIT()
    # test position_embedding
    print(model.position_embedding(197, 768).shape)
    test.testModuleSize(model)
