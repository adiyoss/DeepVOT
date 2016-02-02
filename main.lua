-- for debuing purposes
--require('mobdebug').start()
require('rnn')

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
   cmd:option('-data_type', 'vot', 'the type of the data: lm | dummy | vot')
   cmd:option('-train', 'train.t7', 'the path to the train data')
   cmd:option('-test', 'test.t7', 'the path to the test data')
   cmd:option('-batch_size', 16, 'the mini-batch size')
   cmd:option('-input_dim', 9, 'the input size')   
   
   -- for the model
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')
   -- for the loss
   cmd:option('-loss', 'nll', 'the type of loss function: nll') 
   -- for the train
   cmd:option('-model_type', 'bi-lstm', 'the type of model: bi-lm | bi-lstm | lstm') 
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-plot', true, 'live plot')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: ADAGRAD | SGD')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   -- for the main
   cmd:option('-rho', 50, 'max sequence length')
   cmd:option('-n_epochs', 10, 'max sequence length')   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

dofile('data.lua')
dofile('model.lua')
dofile('loss.lua')
dofile('train.lua')
dofile('test.lua')

local iteration = 1

-- TODO create early stopping criteria !!!
-- training
while iteration<opt.n_epochs do       
  -- train - forward and backprop
  train(train_x, train_y) 
  
  -- increase iteration number
  iteration = iteration + 1  
end

-- 3. evaluate on the test set
local loss = test(test_x, test_y)
print('Loss = ' .. loss)
