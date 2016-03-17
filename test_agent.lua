--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

require 'paths'
require 'image'
local gd = require "gd"
require "initenv"

local opt = require 'opts.opts'
opt.checkpoint_path = 
  '/storage/atari/breakout/DQN3_0_1_breakout_FULL_Y_TEST_TEST'
  --'/storage/atari/breakout/DQN3_0_1_breakout_FULL_Y_FULL_BN'
  --'/storage/atari/breakout/DQN3_0_1_breakout_FULL_Y_TEST'
opt.network = paths.concat(opt.checkpoint_path, 'model.t7')
opt.gif_file= paths.concat(opt.checkpoint_path, 'gifs/test.gif')
--opt.csv_file= paths.concat(opt.checkpoint_path, 'csv/test.csv')
opt.best = false
opt.X11 = false

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
agent.network:evaluate()
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
local win
if opt.X11 then win = image.display({image=screen}) end

print("Started playing...")
local step = 0
local total_reward = 0
-- play one episode (game)
while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    -- choose the best action
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    total_reward = total_reward + reward

    -- display screen
    if opt.X11 then image.display({image=screen, win=win}) end

    io.flush(print(string.format(
      'step: %d, action_index: %d, reward: %f, total_reward: %f, terminal: %s', 
      step, action_index, reward, total_reward, tostring(terminal)))) 
    step = step + 1
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
