--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

require "initenv"
require 'cudnn'

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
  -- fully connected layer
  net:add(nn.Linear(nel, args.n_hid[1]))
  net:add(cudnn.ReLU(true))
  local last_layer_size = args.n_hid[1]

  for i=1,(#args.n_hid-1) do
    -- add Linear layer
    last_layer_size = args.n_hid[i+1]
    net:add(nn.Linear(args.n_hid[i], last_layer_size))
    net:add(cudnn.ReLU(true))
  end

  -- add the last fully connected layer (to actions)
  net:add(nn.Linear(last_layer_size, args.n_actions))

  net:cuda()
  if args.verbose >= 2 then
    print(net)
    print('Convolutional layers flattened output size:', nel)
  end
  return net
end
