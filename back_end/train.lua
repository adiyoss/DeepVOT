require('torch')   -- torch
require('xlua')   -- xlua provides useful tools, like progress bars
require('optim')   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-Directional LSTM')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-batch_size', 64, 'the mini-batch size')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-plot', true, 'live plot')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: ADAGRAD | SGD')   
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   rnn:cuda()
   criterion:cuda()
end
----------------------------------------------------------------------

print '==> defining some tools'
-- classes
classes = {'1','2', '3', '4'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if rnn then
   parameters, gradParameters = rnn:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'
if opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.sgd
elseif opt.optimization == 'ADAGRAD' then
   optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.adagrad
else
   error('unknown optimization method')
end
----------------------------------------------------------------------
print '==> defining training procedure'

function train(train_x, train_y)
    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    rnn:training()
  
    -- shuffle at each epoch
    shuffle = torch.randperm(train_x:size(1))
        
    -- do one epoch
    print('\n==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ']')
    for t = 1,train_x:size(1), opt.batch_size do
      -- disp progress
      xlua.progress(t, train_x:size(1))
    
      -- create mini batch
      local inputs, targets = {}, {}
      
      if opt.data_type == 'lm' then
        for step=1, opt.rho do
            -- a batch of inputs
            inputs[step] = sequence:index(1, offsets)
            -- incement indices
            offsets:add(1)
            for j=1,opt.batch_size do
               if offsets[j] > opt.batch_size then
                  offsets[j] = 1
               end
            end
            targets[step] = sequence:index(1, offsets)
        end
      elseif opt.data_type == 'dummy' or opt.data_type == 'vot' or opt.data_type == 'neg_vot' or opt.data_type == 'multi' then
        for step=1, opt.batch_size do
          -- a batch of inputs
          inputs[step], targets[step] = get_mini_batch(train_x, train_y, opt.rho)
        end
      end      

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)  
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                                              
                       -- reset gradients
                       gradParameters:zero()                       
                       
                       local outputs = rnn:forward(inputs)                        
                       local err = criterion:forward(outputs, targets)
                                              
                       -- 3. backward sequence through rnn (i.e. backprop through time)   
                       local gradOutputs = criterion:backward(outputs, targets)
                       local gradInputs = rnn:backward(inputs, gradOutputs)                      
                       
                       for i=1,#targets do
                          for j=1,opt.rho do
                            -- update confusion                    
                            confusion:add(outputs[i][j], targets[i][j])
                          end
                       end    
                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       err = err/(#inputs)
                       
                       -- update logger/plot
                       -- tracking the gradients
                       gradLogger:add{['% grad norm (train set)'] = torch.norm(gradParameters)}
                       
                       return err, gradParameters
                    end      
      optimMethod(feval, parameters, optimState)   
    end
    -- time taken
    time = sys.clock() - time
    time = time / train_x:size(1)
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    -- print confusion matrix
    print(confusion)
   
    -- update logger/plot
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
    end
    
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end