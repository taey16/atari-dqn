--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]


require "initenv"
local opt = require 'opts.opts'
require 'optim'

local logger_tst = 
  optim.Logger(paths.concat(opt.checkpoint_path,'test.log' ))
local logger_tst_avg = 
  optim.Logger(paths.concat(opt.checkpoint_path,'test_avg.log' ))

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
local win

local screen, reward, terminal = game_env:getState()

io.flush(print("===> Iteration ..", step))
while step < opt.steps do
  step = step + 1
  local action_index = agent:perceive(reward, screen, terminal)
  if not terminal then
    screen, reward, terminal = 
      game_env:step(game_actions[action_index], true)
  else
    if opt.random_starts > 0 then
      screen, reward, terminal = game_env:nextRandomGame()
    else
      screen, reward, terminal = game_env:newGame()
    end
  end

  if X11 then win = image.display({image=screen, win=win}) end

  if step % opt.prog_freq == 0 then
    assert(step==agent.numSteps, 'trainer step: ' .. step ..
        ' & agent.numSteps: ' .. agent.numSteps)
    io.flush(print("Steps: ", step))
    agent:report()
    io.flush(print(string.format(
      '%d, action idx: %d, reward: %f', step, action_index, reward)))
    collectgarbage()
  end

  if step % 1000 == 0 then collectgarbage() end
  
  -- eval
  if step % opt.eval_freq == 0 and step > learn_start then
    screen, reward, terminal = game_env:newGame()
    local eval_time = sys.clock()

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    for estep=1,opt.eval_steps do
      local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
      -- Play game in test mode (episodes don't end when losing a life)
      screen, reward, terminal = game_env:step(game_actions[action_index])

      if X11 then win = image.display({image=screen, win=win}) end
      if estep % 1000 == 0 then collectgarbage() end

      episode_reward = episode_reward + reward
      if reward ~= 0 then
         nrewards = nrewards + 1
      end

      if terminal then
        total_reward = total_reward + episode_reward
        episode_reward = 0
        nepisodes = nepisodes + 1
        screen, reward, terminal = game_env:nextRandomGame()
      end
    end

    eval_time = sys.clock() - eval_time
    start_time = start_time + eval_time
    agent:compute_validation_statistics()
    local ind = #reward_history+1
    total_reward = total_reward/math.max(1, nepisodes)

    if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
      agent.best_network = agent.network:clone()
    end

    if agent.v_avg then
      v_history[ind] = agent.v_avg
      td_history[ind] = agent.tderr_avg
      qmax_history[ind] = agent.q_max
      logger_tst_avg:add{
        ['step'] = step,
        ['v_avg'] = agent.v_avg,
        ['tderr_avg'] = agent.tderr_avg,
        ['g_max'] = agent.q_max -- action value Q
      }
    end
    io.flush(print(
      "V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind]))

    reward_history[ind]= total_reward
    reward_counts[ind] = nrewards
    episode_counts[ind]= nepisodes

    logger_tst:add{
      ['step'] = step,
      ['total_reward'] = total_reward,
      ['reward_counts'] = nrewards,
      ['epsoid_counts'] = nepisodes
    }

    time_history[ind+1] = sys.clock() - start_time
    local time_dif = time_history[ind+1] - time_history[ind]
    local training_rate = opt.actrep*opt.eval_freq/time_dif

    io.flush(print(string.format(
      '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
      'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
      'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
      step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
      training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
      nepisodes, nrewards)))
  end -- end of eval

  if step % opt.save_freq == 0 or step == opt.steps then
    local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
      agent.valid_s2, agent.valid_term
    agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
      agent.valid_term = nil, nil, nil, nil, nil, nil, nil
    local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
      agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
    agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
      agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

    local filename = paths.concat(opt.checkpoint_path, 'model')
    if opt.save_versions > 0 then
      filename = filename .. "_" .. math.floor(step / opt.save_versions)
    end
    filename = filename
    torch.save(filename .. ".t7", {agent = agent,
                model = agent.network,
                best_model = agent.best_network,
                reward_history = reward_history,
                reward_counts = reward_counts,
                episode_counts = episode_counts,
                time_history = time_history,
                v_history = v_history,
                td_history = td_history,
                qmax_history = qmax_history,
                arguments=opt})
    if opt.saveNetworkParams then
      local nets = {network=w:clone():float()}
      torch.save(filename..'.params.t7', nets, 'ascii')
    end
    io.flush(print(string.format('===> save done in %s', filename)))
    agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
      agent.valid_term = s, a, r, s2, term
    agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
      agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
    io.flush(print('Saved:', filename .. '.t7'))
    collectgarbage()
  end -- end of save
end -- end of main loop
