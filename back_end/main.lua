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
   cmd:option('-data_type', 'multi', 'the type of the data: lm | dummy | vot | neg_vot | multi')
   cmd:option('-train', 'train.t7', 'the path to the train data')
   cmd:option('-test', 'test.t7', 'the path to the test data')
   cmd:option('-batch_size', 1, 'the mini-batch size')
   cmd:option('-input_dim', 63, 'the input size')  
   cmd:option('-val_percentage', 0.15, 'the percentage of exampels to be considered as validation set from the training set')
   
   -- for the model
   cmd:option('-model_type', 'lstm', 'the type of model: bi-lm | bi-lstm | lstm') 
   cmd:option('-output_dim', 4, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')
   cmd:option('-dropout', 0.8, 'dropout rate')
   
   -- for the loss
   cmd:option('-loss', 'nll', 'the type of loss function: nll') 
   
   -- for the train
   cmd:option('-save', 'results/', 'subdirectory to save/log experiments in')
   cmd:option('-plot', true, 'live plot')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: ADAGRAD | SGD')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   
   -- for the main
   cmd:option('-rho', 10, 'max sequence length')
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
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
validationLogger = optim.Logger(paths.concat(opt.save, 'validate.log'))
gradLogger = optim.Logger(paths.concat(opt.save, 'grad.log'))
paramsLogger = io.open(paths.concat(opt.save, 'params.log'), 'w')

-- save cmd parameters
for key, value in pairs(opt) do
  paramsLogger:write(key .. ': ' .. tostring(value) .. '\n')
end
paramsLogger:close()

dofile('data.lua')
dofile('model.lua')
dofile('loss.lua')
dofile('train.lua')
dofile('test.lua')
dofile('validate.lua')

local iteration = 1
local best_loss = 100000
local loss = 0
--[[
local train_balance = torch.sum(train_y-1)/train_y:size(1)
local val_balance = torch.sum(val_y-1)/val_y:size(1)
local test_balance = torch.sum(test_y-1)/test_y:size(1)
]]--
-- data statistics
print('\n==> data statistics: ')
print('==> number of training examples: ' .. #train_data)
print('==> number of validation examples: ' .. #val_data)
print('==> number of test examples: ' .. #test_data)
--[[
print('\n==> training set balance: first label: ' .. train_balance .. ', second label: ' .. (1-train_balance))
print('==> validation set balance: first label: ' .. val_balance .. ', second label: ' .. (1-val_balance))
print('==> test set balance: first label: ' .. test_balance .. ', second label: ' .. (1-test_balance))
]]--

-- gets the first loss on the validation set
loss = validate(val_data)
print('==> validation loss: ' .. loss)

-- training
while loss < best_loss or iteration < opt.patience do       
  
  -- train - forward and backprop
  train(train_data) 
    
  -- validate  
  loss = validate(val_data)
  print('\n==> validation loss: ' .. loss)
  
  -- for early stopping criteria
  if loss >= best_loss then 
    -- increase iteration number
    iteration = iteration + 1
    print('\n========================================')
    print('==> Loss did not improved, iteration: ' .. iteration)
    print('========================================\n')
  else
    -- update the best loss value
    best_loss = loss
    
    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)    
    torch.save(filename, rnn)
    iteration = 1
  end
end

-- evaluate on the test set
local test_loss = test(test_data)
print('\n============ EVALUATING ON TEST SET ============')
print('Loss = ' .. test_loss .. '\n')
