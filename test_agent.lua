--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"

if not dqn then
    require "initenv"
end

local env_name = 'breakout'
local rom_path = '/works/DeepMind-Atari-Deep-Q-Learner/roms'
local actrep = 4
local checkpoint_agent_filename = string.format('DQN3_0_1_%s_FULL_Y.t7', env_name)
local agent_params = 'lr=0.00025,ep=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network="convnet_atari3",preproc="net_downsample_2x_full_y",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1'
local gif_filename = string.format('gifs/%s_%s.gif', env_name, checkpoint_agent_filename)


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', 'alewrap', 'name of training framework')
cmd:option('-env', env_name, 'name of environment to use')
cmd:option('-game_path', rom_path, 'path to environment file (ROM)')
cmd:option('-env_params', 'useRGB=true', 'string of environment parameters')
cmd:option('-pool_frms', 'type=\"max\",size=2',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', actrep, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', checkpoint_agent_filename, 
  'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', agent_params, 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', gif_filename, 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = game_env:newGame()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = image.display({image=screen})

print("Started playing...")

-- play one episode (game)
while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    
    -- choose the best action
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)

    -- display screen
    image.display({image=screen, win=win})

    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im

end

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
