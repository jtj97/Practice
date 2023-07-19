import torch

def testModuleSize(testModule, tensorChannel = 3, tensorSize = 224):
    x = torch.zeros((32, tensorChannel, tensorSize, tensorSize))  # minibatch size 32, image size [tensorChannel, tensorSize, tensorSize]
    output = testModule(x)
    print(output.size())

def testSequence(testModule, patch_num = 196, dim = 768):
    x = torch.zeros((32, patch_num, dim))
    output = testModule(x)
    # input(32, 196, 768) --> attn: (32, 192, 196)
    print(output.size())
