import torch.nn as nn

'''
           patch                                proj
(N,C,H,W)-------->(N, H * W / P^2, P * P * C)--------->(N, H * W / P^2, D), D is the embedding_dim
'''
class PatchEmbedding(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, channels = 3, embedding_dim = 768, norm_layer = None, flatten = True):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.grid_size = self.image_size // patch_size # patch num in one dimension
        self.num_patches = self.grid_size * self.grid_size
        self.flatten =  flatten
        
        # use conv to patchify
        self.proj = nn.Conv2d(channels, embedding_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.image_size and W == self.image_size
        
        x = self.proj(x) # (N, C, H, W)---->(N, D, H/P, W/P)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # (N, D, H/P, W/P) ----> (N, HW/P^2, D)
            
        x = self.norm(x)
        
        return x
    
if __name__ == '__main__':
    import test.testMain as test
    module = PatchEmbedding()
    # input(32, 3, 224, 224), grid_size = 224/16 = 14 ,patch_num = 14 * 14 = 196, embedding_dim = 768.so except output = (32,196,768)
    test.testModuleSize(module)
    
