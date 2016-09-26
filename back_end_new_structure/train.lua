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


if opt.output_dim == 4 then  
  classes = {'1', '2', '3', '4'}
elseif opt.output_dim == 2 then
  classes = {'1', '2'}
else
  classes = {'1', '2', '3', '4'}
end

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
elseif opt.optimization == 'ADAM' then
   optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.adam
else
   error('unknown optimization method')
end
----------------------------------------------------------------------
print '==> defining training procedure'
    
function train(data, logger)
    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    rnn:training()
  
    -- do one epoch
    print('\n==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    for t = 1,#data do
      -- disp progress
      xlua.progress(t, #data)
      
      -- get input and output
      local targets = data[t][{ {}, {}, 1 }] + 1 -- torch start counting from 1
      local inputs = data[t][{ {}, {}, {2, data[t]:size(3)}}]
      
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
                       
                       for i=1,targets:size(1) do
                          -- update confusion                    
                          confusion:add(outputs[i][1], targets[i][1])
                       end
                       
                       -- gradient clipping
                       gradParameters:clamp(-5, 5)
                       
                       -- normalize gradients and f(X)
                       -- gradParameters:div(inputs:size(1))
                       err = err/(inputs:size(1))
                       
                       -- update logger/plot
                       -- tracking the gradients
                       gradLogger:add{['% grad norm (train set)'] = torch.norm(gradParameters)}
                       
                       return err, gradParameters
                    end      
      optimMethod(feval, parameters, optimState)   
    end
    
    -- time taken
    time = sys.clock() - time
    time = time / #data
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    -- print confusion matrix
    print(confusion)
   
    -- update logger/plot
    logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    if opt.plot then
      logger:style{['% mean class accuracy (train set)'] = '-'}
      logger:plot()
    end
    
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end