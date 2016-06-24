require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> defining validate procedure'

-- test function
function validate(data)  
  
  -- local vars
  local time = sys.clock()

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  rnn:evaluate()
  
  -- validate over validation data
  print('\n==> validating on validation set:')
  local err = 0
  local count = 1
  for i=1,#data do    
    -- disp progress
    xlua.progress(i, #data)
    count = count + data[i]:size(1) 
    
    local input, target = {}, {}
    for t=1, data[i]:size(1) do
      table.insert(input, data[i][t][{{2, (opt.input_dim+1)}}])
      table.insert(target, (data[i][t][1] + 1)) -- torch starts counting from 1
    end
    
    -- validate sample
    local output = rnn:forward(input)
    err = err + criterion:forward(output, target)
    
    for i=1,#target do
      -- update confusion                    
      confusion:add(output[i], target[i])
    end    
  end

  -- timing
  time = sys.clock() - time
  time = time / count
  print("==> time to validate 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)  
  -- update log/plot
  validationLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  
  -- initialize confusion matrix for next epoch
  confusion:zero()
  return err / #data
end