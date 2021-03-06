
require 'nn'
require 'cunn'
require 'cudnn'

-- Delving Deep into Rectifiers: 
-- Suupassing Human-Level Performance on ImageNet Classification, 2014
-- A sufficient condition is:
-- \frac{1}{2} n_l \text{Var}[w_l] = 1, \forall l. (1)
-- where n_l = k^2 * c
-- (1) leads to a zero-mean Gaussian distribution whos std. is \sqrt{2/n_l}
-- bias is zeros
function MSRinit(net)
  local function init(module_type)
    for k, v in pairs(net:findModules(module_type)) do
      -- n = k^2 * c
      local n = v.kW * v.kH * v.nOutputPlane
      -- \sqrt{2/n}
      v.weight:normal(0,math.sqrt(2/n))
      -- As FAIR did (fb.ResNet.torch)
      -- All of the cudnn.SpatialConv. layers are followered by 
      -- cudnn.SpaatialBatchNorm.
      -- which means we did not need to set a bias vector in conv. layers
      if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
      else
        v.bias:zero()
      end
    end
  end
  local function BNinit(module_type)
    for k,v in pairs(net:findModules(module_type)) do
      v.weight:fill(1)
      v.bias:zero()
    end
  end
  local function Linearinit(module_type)
    for k,v in pairs(net:findModules(module_type)) do
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
  init'nn.SpatialConvolutionMM'
  init'cudnn.SpatialConvolution'
  BNinit'nn.SpatialBatchNormalization'
  BNinit'nn.BatchNormalization'
  BNinit'cudnn.SpatialBatchNormalization'
  BNinit'cudnn.BatchNormalization'
  Linearinit'nn.Linear'
end

