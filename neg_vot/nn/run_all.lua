----------------------------------------------------------------------
----------------------------------------------------------------------
--require('mobdebug').start()
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Neg VOT')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1224, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 4, 'number of threads')
cmd:option('-type', 'double', 'the data type: double')
cmd:option('-n_epochs', 50, 'the number of epochs')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')

-- data:
cmd:option('-input_dim', 900, 'The input size') 

-- model:
cmd:option('-model', 'fc', 'type of model to construct: fc | conv')
cmd:option('-num_class', 2, 'the number of classes')

-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | margin')

-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'ADAGRAD', 'optimization method: SGD | ADAGRAD')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')

cmd:text()
opt = cmd:parse(arg or {})

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'
----------------------------------------------------------------------
i = 0
while i<opt.n_epochs do
   train()
   i = i + 1
end
-- evaluating of test set
test()

-- save/log current net
local filename = paths.concat(opt.save, 'model.net')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving model to '..filename)
torch.save(filename, model)
print('==> Done.')