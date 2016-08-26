--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

require "initenv"
require 'cudnn'
require 'GradientRescale'
local DualAggregator = require 'DuelAggregator'

function create_network(args)
  local net = nn.Sequential()
  net:add(nn.Reshape(unpack(args.input_dims)))

  net:add(cudnn.SpatialConvolution(
    args.hist_len*args.ncols, args.n_units[1],
    args.filter_size[1], args.filter_size[1],
    args.filter_stride[1], args.filter_stride[1],1,1))
  net:add(cudnn.ReLU(true))

  -- Add convolutional layers
  for i=1,(#args.n_units-1) do
    -- second convolutional layer
    net:add(cudnn.SpatialConvolution(
      args.n_units[i], args.n_units[i+1],
      args.filter_size[i+1], args.filter_size[i+1],
      args.filter_stride[i+1], args.filter_stride[i+1],0,0))
    net:add(cudnn.ReLU(true))
  end
  local nel = net:cuda():forward(
    torch.zeros(1,unpack(args.input_dims)):cuda()):nElement()
  -- reshape all feature planes into a vector per example
  net:add(nn.Reshape(nel))

  local head = nn.Sequential()
  head:add(nn.GradientRescale(1/math.sqrt(2), true))

  -- Value approximator V^(s)
  local valueNetStream = nn.Sequential()
  valueNetStream:add(nn.Linear(nel, args.n_hid[1]))
  valueNetStream:add(cudnn.ReLU(true))
  local last_layer_size = args.n_hid[1]
  --[[
  for i=1,(#args.n_hid-1) do
    last_layer_size = args.n_hid[i+1]
    valueNetStream:add(nn.Linear(args.n_hid[i], last_layer_size))
    valueNetStream:add(cudnn.ReLU(true))
  end
  --]]
  valueNetStream:add(nn.Linear(last_layer_size, 1))

  -- Advantage approximator A^(s,a)
  local advantageStream = nn.Sequential()
  advantageStream:add(nn.Linear(nel, args.n_hid[1]))
  advantageStream:add(cudnn.ReLU(true))
  local last_layer_size = args.n_hid[1]
  for i=1,(#args.n_hid-1) do
    last_layer_size = args.n_hid[i+1]
    advantageStream:add(nn.Linear(args.n_hid[i], last_layer_size))
    advantageStream:add(cudnn.ReLU(true))
  end
  advantageStream:add(nn.Linear(last_layer_size, args.n_actions))

  local streams = nn.ConcatTable()
  streams:add(valueNetStream)
  streams:add(advantageStream)

  head:add(streams)
  head:add(DualAggregator(args.n_actions))

  net:add(head)

  net:cuda()
  if args.verbose >= 2 then
    print(net)
    print('Convolutional layers flattened output size:', nel)
  end
  return net
end
