-- for debuing purposes
--require('mobdebug').start()
require('torch')
require('nn')
require('rnn')
require('optim')

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '\n==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-Directional LSTM')
   cmd:text()
   cmd:text('Options:')
   -- for the data
   cmd:option('-data_type', 'multi', 'the type of the data: dummy | multi | vot')
   cmd:option('-train', 'train.t7', 'the path to the train data')
   cmd:option('-test', 'test.t7', 'the path to the test data')
   cmd:option('-val', 'val.t7', 'use it only with vot')   
   cmd:option('-batch_size', 1, 'the mini-batch size')
   cmd:option('-input_dim', 63, 'the input size')  
   cmd:option('-val_percentage', 0.2, 'the percentage of exampels to be considered as validation set from the training set')
   
   -- for the model
   cmd:option('-output_dim', 4, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')
   cmd:option('-dropout', 0.2, 'dropout rate')
   
   -- for the loss
   cmd:option('-loss', 'nll', 'the type of loss function: nll') 
   
   -- for the train
   cmd:option('-save', 'results/deep_vot/', 'subdirectory to save/log experiments in')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: ADAGRAD | SGD')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   cmd:option('-plot', true, 'live plot')
   
   -- for the main
   cmd:option('-patience', 5, 'the number of epochs to be patient before early stopping')
   cmd:option('-seed', 1245, 'the starter seed, for randomness')
   cmd:option('-threads', 4, 'the number of threads') -- check this
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Log results to files
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

dofile('data.lua')
dofile('model.lua')
dofile('loss.lua')
dofile('train.lua')
dofile('validate.lua')

local iteration = 1
local best_loss = 100000
local loss = 0

-- data statistics
print('\n==> data statistics: ')
print('==> number of training examples: ' .. #train_data)
print('==> number of validation examples: ' .. #val_data)
print('==> number of test examples: ' .. #test_data)

rnn = torch.load(paths.concat(opt.save, 'model.net'))

-- evaluate on the test set
local test_loss = validate(test_data, testLogger)
print('\n============ EVALUATING ON TEST SET ============')
print('Loss = ' .. test_loss .. '\n')
