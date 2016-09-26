-- for debuing purposes
--require('mobdebug').start()
require 'xlua'    -- xlua provides useful tools, like progress bars
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
   cmd:option('-test', 'data/natalia_pos/test.t7', 'the path to the test data')
   
   cmd:option('-model', 'results/natalia_pos/model.net', 'the path to the model')
   cmd:option('-output_file', 'results/natalia_pos/pred.txt', 'the output file to pritn the predictions')
      
   cmd:option('-seed', 1245, 'the starter seed, for randomness')
   cmd:option('-threads', 4, 'the number of threads') -- check this
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
local function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

local function argmax_2D(matrix)
   local nRows = matrix:size(1)
   local result = torch.Tensor(nRows)
   for i = 1, nRows do
      result[i] = argmax_1D(matrix[i])
   end
   return result
end


torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
  
-- load the data
test_data = torch.load(opt.test)

-- load the model
model = torch.load(opt.model)

out_fid = io.open(opt.output_file, 'w')
for i=1, #test_data[1] do  
  -- disp progress
  xlua.progress(i, #test_data[1])
  
  local output = model:forward(test_data[1][i])    
  local re_output = output:reshape(output:size(1), output:size(3))
  local y_hat = argmax_2D(re_output)
  local pred = {}
  local flag = false
  local onset = 1
  for j=2, y_hat:size(1) do
    if y_hat[j-1] ~= y_hat[j] then
      if flag == true then
        table.insert(pred, {onset, j})
        flag = false
      else
        flag = true
        onset = j
      end
    end
  end
  
  if flag == true then
    table.insert(pred, {onset, y_hat:size(1)})    
  end
  
  local max_idx = 1
  local max_length = pred[1][2] - pred[1][1]
  for j=1, #pred do
    local tmp = pred[j][2] - pred[j][1]
    if tmp > max_length then
      max_idx = j
      max_length = tmp
    end
  end  

  out_fid:write(test_data[3][i] .. ' ' .. tostring(pred[max_idx][1]) .. ' ' .. tostring(pred[max_idx][2]) .. '\n')
end
out_fid.close()



