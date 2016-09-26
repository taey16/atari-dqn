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
  --'space_invaders'
  --'seaquest'
  --'breakout'
  --'pong'
  'qbert'
  --'beam_rider'
  --'enduro'
opt.checkpoint_path = 
  ('/storage/atari/%s/DQN3_0_1_FULL_Y_DUAL_reinforce_lr25e-6'):format( opt.env_name, opt.env_name )
  --('/storage/atari/%s/DQN3_0_1_FULL_Y_DUAL'):format( opt.env_name, opt.env_name )
  --('/storage/atari/%s/DQN3_0_1_FULL_Y'):format( opt.env_name, opt.env_name )
  --('/storage/atari/%s/DQN3_0_1_FULL_Y_reinforce_lr25e-6'):format( opt.env_name, opt.env_name )
opt.network = paths.concat(opt.checkpoint_path, 'model.t7')
opt.gif_file= paths.concat(opt.checkpoint_path, 'gifs/test.gif')
opt.csv_file= paths.concat(opt.checkpoint_path, 'logs/test.log')
opt.best = true
opt.X11 = false
epsilon = 0.00


--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
agent.network:evaluate()

print("Started playing...")
local total_reward_per_episoid = 0
local total_reward_per_episoid_history = {}
local mean_reward = 0
local n_episoid = 5
for episoid_id = 1,n_episoid do
  local step = 1
  local avg_best_q = 0
  local total_reward = 0

  -- start a new game
  print(string.format('Start a new %d th game...', episoid_id))
  local screen, reward, terminal = game_env:newGame()
  -- compress screen to JPEG with 100% quality
  local jpg = image.compressJPG(screen:squeeze(), 100)
  -- create gd image from JPEG string
  local im = gd.createFromJpegStr(jpg:storage():string())
  -- convert truecolor to palette
  im:trueColorToPalette(false, 256)
  -- file names from command line
  local gif_filename = string.format('%s_%02d.gif', opt.gif_file, episoid_id)
  -- write GIF header, use global palette and infinite looping
  im:gifAnimBegin(gif_filename, true, 0)
  -- write first frame
  im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
  local csv_filename = string.format('%s_%02d.log', opt.csv_file, episoid_id)
  local logger_tst = optim.Logger(csv_filename)

  -- remember the image and show it first
  local previm = im
  local win
  if opt.X11 then win = image.display({image=screen}) end

  -- play one episode (game)
  while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    local is_exploitation
    -- choose the best action
    local action_index, is_exploitation = 
      agent:perceive(reward, screen, terminal, true, epsilon)
    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    total_reward = total_reward + reward
    avg_best_q = avg_best_q + agent.bestq
    step = step + 1

    -- display screen
    if opt.X11 then image.display({image=screen, win=win}) end
    io.flush(print(string.format(
      'step: %d, rewards: %f, total_rewards: %f, bestQ: %f, avg-bestQ: %f, exploidation: %d', 
      step, reward, total_reward, agent.bestq, avg_best_q / step, is_exploitation))) 
    logger_tst:add{
      ['step'] = step,
      ['reward'] = reward,
      ['total_reward'] = total_reward,
      ['bestq'] = agent.bestq,
      ['avg_bestq'] = avg_best_q / step,
      ['exploidation'] = is_exploitation
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
  total_reward_per_episoid = total_reward_per_episoid + total_reward
  mean_reward = total_reward_per_episoid / episoid_id
  table.insert(total_reward_per_episoid_history, total_reward)

  print(string.format(
    '%d th game: total_reward: %d, avg_total_reward: %f', 
    episoid_id, total_reward, mean_reward))

  -- end GIF animation and close CSV file
  gd.gifAnimEnd(gif_filename)
  print("Finished game")
  print(string.format('saved gif: %s', gif_filename))
  print(string.format('saved csv: %s', csv_filename))
  io.flush()
end

local std_reward = 0
for ep=1,#total_reward_per_episoid_history do
  std_reward = std_reward + math.abs(mean_reward - total_reward_per_episoid_history[ep]) 
end
print(string.format('std_reward: %f', std_reward / #total_reward_per_episoid_history))
print('Done')
