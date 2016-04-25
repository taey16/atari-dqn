
require 'paths'

local env_name = 
  --'space_invaders'
  --'seaquest'
  --'breakout'
  --'pong'
  'qbert'

local priority_ratio = 0.25
local agent_filename = 'NeuralQLearner'
local agent_params = 
  -- explict reinforce
  --string.format( 'lr=0.000025,eq=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network=\"convnet_atari3\",preproc=\"net_downsample_2x_full_y\",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1,priority_ratio=%f', priority_ratio)
  -- initial lr=0.00025
  string.format('lr=0.00025,eq=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network=\"convnet_atari3\",preproc=\"net_downsample_2x_full_y\",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1,priority_ratio=%f', priority_ratio)

-- frame-skipping technique
local actrep = 4

local reinforce = 
  string.format(
    '')
    --'/storage/atari/%s/DQN3_0_1_breakout_FULL_Y_FULL/model.t7', env_name)
    --'/storage/atari/%s/DQN3_0_1_seaquest_FULL_Y_FULL/model.t7', env_name)
    --'/storage/atari/%s/DQN3_0_1_FULL_Y/model.t7', env_name)
local best = true

local prog_freq = 50000--10000 --frequency of progress output
local save_freq = 250000--125000 --the model is saved every save_freq steps
local eval_freq = 250000 --frequency of greedy evaluation
local eval_steps= 135000--125000 --number of evaluation steps

local checkpoint_agent_name = 
  'DQN3_0_1_FULL_Y_reinforce_lr25e-6'
  --'DQN3_0_1_FULL_Y_FULL_reinforce_lr25e-6'
  --'DQN3_0_1_FULL_Y_FULL_reinforce'
  --'DQN3_0_1_FULL_Y'
  --string.format('DQN3_0_1_FULL_Y_priority%.2f', priority_ratio)
local checkpoint_path = 
  string.format('/storage/atari/%s/%s', env_name, checkpoint_agent_name)
local X11 = false

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', 'alewrap', 'name of training framework')
cmd:option('-env', env_name, 'name of environment to use')
cmd:option('-game_path', 'roms', 'path to environment file (ROM)')
cmd:option('-env_params', 'useRGB=true', 'string of environment parameters')
cmd:option('-pool_frms', 'type=\"max\",size=2',
  'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', actrep, 
  'how many times to repeat action ' ..
  '(Repeat each action selected by the agent this many time. ' ..
  'Using a value of 4 results in the agent seeing only every 4th input frame.')
cmd:option('-random_starts', 30, 'play action 0 between 1 and random_starts ' ..
  'number of times at the start of each training episode')

cmd:option('-network', reinforce, 'reload pretrained network')
cmd:option('-agent', agent_filename, 'name of agent file to use')
cmd:option('-agent_params', agent_params, 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
  'saves the agent network in a separate file for each evaluation')
cmd:option('-prog_freq', prog_freq, 'frequency of progress output')
cmd:option('-save_freq', save_freq, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', eval_freq, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 50000000, 'number of training steps to perform')
cmd:option('-eval_steps',eval_steps, 'number of evaluation steps')

cmd:option('-verbose', 2,
  'the higher the level, the more information is printed to screen')
cmd:option('-threads', 4, 'number of BLAS threads')
cmd:option('-gpus', {1}, 'gpu flag')
cmd:option('-checkpoint_path', checkpoint_path, 
  'filename used for saving network and training history')
cmd:option('--X11', X11, 'use X11 or not')

-- for test
cmd:option('-best', best, 'select best_model in reinforce stage')

cmd:text()

local opts = cmd:parse(arg)
os.execute('mkdir -p '..opts.checkpoint_path)
print('===> Saving everything to: '..opts.checkpoint_path)
os.execute(string.format('mkdir -p %s/gifs',opts.checkpoint_path))
os.execute(string.format('mkdir -p %s/logs',opts.checkpoint_path))
os.execute(string.format('cp opts/opts.lua %s', opts.checkpoint_path))
os.execute(string.format('cp *.lua %s', opts.checkpoint_path))
io.flush()

return opts

