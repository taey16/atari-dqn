--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

require 'paths'
require 'image'
require 'optim'
local gd = require "gd"
require "initenv"

local opt = require 'opts.opts'
opt.env_name = 
  --'seaquest'
  'breakout'
opt.checkpoint_path = 
  ('/storage/atari/%s/DQN3_0_1_%s_FULL_Y_FULL'):format( opt.env_name, opt.env_name )
opt.network = paths.concat(opt.checkpoint_path, 'model.t7')
opt.gif_file= paths.concat(opt.checkpoint_path, 'gifs/test.gif')
opt.csv_file= paths.concat(opt.checkpoint_path, 'logs/test.log')
local logger_tst = optim.Logger(opt.csv_file)
opt.best = true
opt.X11 = false


--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
-- start a new game
local screen, reward, terminal = game_env:newGame()
agent.network:evaluate()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)
-- file names from command line
local gif_filename = opt.gif_file
-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win
if opt.X11 then win = image.display({image=screen}) end

print("Started playing...")
local step = 1
local total_reward = 0
local avg_best_q = 0
-- play one episode (game)
while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    -- choose the best action
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    total_reward = total_reward + reward
    avg_best_q = avg_best_q + agent.bestq
    step = step + 1

    -- display screen
    if opt.X11 then image.display({image=screen, win=win}) end
    io.flush(print(string.format(
      'step: %d, rewards: %f, total_rewards: %f, bestQ: %f, avg-bestQ: %f', 
      step, reward, total_reward, agent.bestq, avg_best_q / step))) 
    logger_tst:add{
      ['step'] = step,
      ['reward'] = reward,
      ['total_reward'] = total_reward,
      ['bestq'] = agent.bestq,
      ['avg_bestq'] = avg_best_q / step
    }

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
print(string.format('saved gif: %s', opt.gif_file))
print(string.format('saved csv: %s', opt.csv_file))
io.flush()
