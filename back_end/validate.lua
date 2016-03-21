require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> defining validate procedure'

-- test function
function validate(inputs, targets)  
  
  -- local vars
  local time = sys.clock()

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  rnn:evaluate()

  -- test over test data
  print('\n==> validating on validation set:')
  local err = 0
  for i=1,inputs:size(1) do
    -- disp progress
    xlua.progress(i, inputs:size(1))
    
    local input, target = {}, {}
    table.insert(input, inputs[i])
    table.insert(target, targets[i])
    
    -- test sample
    local output = rnn:forward(input)
    err = err + criterion:forward(output, target)
    
    confusion:add(output[1], target[1])
  end
  
  -- timing
  time = sys.clock() - time
  time = time / inputs:size(1)
  print("==> time to validate 1 sample = " .. (time*1000) .. 'ms')
  
  -- print confusion matrix
  print(confusion)
  
  -- update log/plot
  validationLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  
  -- initialize confusion matrix for next epoch
  confusion:zero()
  
  return err / inputs:size(1)
end