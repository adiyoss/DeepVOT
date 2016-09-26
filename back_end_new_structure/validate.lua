require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> defining validate procedure'

-- test function
function validate(data, logger)  
  
  -- local vars
  local time = sys.clock()

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  rnn:evaluate()
  
  -- validate over validation data
  print('\n==> evaluate on validation set:')
  local err = 0
  for i=1,#data do    
    -- disp progress
    xlua.progress(i, #data)
    
    -- get input and output
    local target = data[i][{ {}, {}, 1 }] + 1 -- torch start counting from 1
    local input = data[i][{ {}, {}, {2, data[i]:size(3)}}]
    
    -- validate sample
    local output = rnn:forward(input)
    err = err + criterion:forward(output, target)
    
    for i=1,target:size(1) do
      -- update confusion                    
      confusion:add(output[i][1], target[i][1])
    end    
  end

  -- timing
  time = sys.clock() - time
  time = time / #data
  print("==> time to evaluate 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)  
  -- update log/plot
  logger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  
  -- initialize confusion matrix for next epoch
  confusion:zero()
  return err / #data
end