import re

def build_aurora2w():
    with open('models/PIDNet/models/pidnet.py', 'r') as f:
        code = f.read()
    
    # 1. Add imports
    imports_to_add = """import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.ops import DeformConv2d
from .idfa import IDFAModule
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
"""
    code = re.sub(r'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport time\nfrom .model_utils import .*', imports_to_add, code, flags=re.MULTILINE)
    
    # 2. Add DeformBasicBlock and DeformSequential
    deform_classes = """
class DeformSequential(nn.Sequential):
    def forward(self, x, offset):
        for module in self:
            x = module(x, offset)
        return x

class DeformBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(DeformBasicBlock, self).__init__()
        self.conv1 = DeformConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x, offset):
        residual = x

        out = self.conv1(x, offset)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class PIDNet("""
    code = code.replace("class PIDNet(", deform_classes)
    
    # 3. Rename PIDNet to AuRoRA2W
    code = code.replace("class PIDNet(nn.Module):", "class AuRoRA2W(nn.Module):")
    code = code.replace("super(PIDNet, self).__init__()", "super(AuRoRA2W, self).__init__()")
    code = code.replace("PIDNet(", "AuRoRA2W(")
    
    # 4. Modify __init__ to instantiate IDFA and use DeformBasicBlock for P branch
    idfa_init = """        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # IDFA Module for P branch geometric compensation
        self.idfa = IDFAModule(in_channels=planes * 2, kernel_size=3)
        """
    code = code.replace("        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)\n        ", idfa_init)
    
    layer_replacement = """        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_deform_layer(DeformBasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_deform_layer(DeformBasicBlock, planes * 2, planes * 2, m)"""
    
    code = re.sub(r'        self.pag3 = PagFM\(planes \* 2, planes\)\n        self.pag4 = PagFM\(planes \* 2, planes\)\n\n        self.layer3_ = self._make_layer\(BasicBlock, planes \* 2, planes \* 2, m\)\n        self.layer4_ = self._make_layer\(BasicBlock, planes \* 2, planes \* 2, m\)', layer_replacement, code)
    
    # 5. Add _make_deform_layer
    make_deform_layer = """
    def _make_deform_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return DeformSequential(*layers)

    def _make_layer("""
    code = code.replace("    def _make_layer(", make_deform_layer)
    
    # 6. Modify forward pass to inject IDFA offsets
    forward_def = """    def forward(self, x, roll_angles=None):
        if roll_angles is None:
            roll_angles = torch.zeros(x.size(0), device=x.device)

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        
        # Start P Branch (Spatial details)
        x_ = x
        
        # Calculate Deformable Offsets via IDFA
        offsets = self.idfa(x_, roll_angles)
        
        x_ = self.layer3_(x_, offsets)"""
        
    code = re.sub(r'    def forward\(self, x\):\n\n        width_output = x.shape\[-1\] // 8\n        height_output = x.shape\[-2\] // 8\n\n        x = self.conv1\(x\)\n        x = self.layer1\(x\)\n        x = self.relu\(self.layer2\(self.relu\(x\)\)\)\n        x_ = self.layer3_\(x\)', forward_def, code)
    
    code = code.replace("x_ = self.layer4_(self.relu(x_))", "x_ = self.layer4_(self.relu(x_), offsets)")
    
    with open('models/PIDNet/models/aurora2w.py', 'w') as f:
        f.write(code)

if __name__ == '__main__':
    build_aurora2w()
