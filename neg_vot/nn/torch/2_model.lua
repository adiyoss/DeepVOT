----------------------------------------------------------------------
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Build the model')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'fc', 'type of model to construct: fc | conv')
   cmd:option('-input_dim', 10, 'the number of inputs')
   cmd:option('-num_class', 2, 'the number of classes')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
noutputs = opt.num_class

----------------------------------------------------------------------
print '==> construct model'
if opt.model == 'fc' then
   hidden_dim = 512
   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Linear(opt.input_dim, hidden_dim))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.8))
   model:add(nn.Linear(hidden_dim, hidden_dim))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.8))
   model:add(nn.Linear(hidden_dim, hidden_dim))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.8))
   model:add(nn.Linear(hidden_dim, noutputs))
   
elseif opt.model == 'conv' then
  
  print('Not yet')
else
   error('unknown -model')
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
----------------------------------------------------------------------
