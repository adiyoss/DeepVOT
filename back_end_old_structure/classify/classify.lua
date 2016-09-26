-- for debuing purposes
--require('mobdebug').start()
require('torch')
require('nn')
require('rnn')

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '\n==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-Directional LSTM - classify')
   cmd:text()
   cmd:text('Options:')

   cmd:option('-data_type', 'vot', 'the type of the data: dummy | vot')
   cmd:option('-test', 'prevoiced_natalia_test.t7', 'the path to the test data')
   cmd:option('-input_dim', 63, 'the input size')
   cmd:option('-model_path', '../results/anamda_2_layers_voiceless_drop_0.3/model.net', 'the path to the model') 
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')
   cmd:option('-dump_dir', 'log', 'the path to save the prediction files')
  
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

-- classify function
function classify(inputs)  
  -- local vars
  local time = sys.clock()
  local predictions = {}
  
  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  rnn:evaluate()

  -- predict over test data
  print('==> classifing:')
  for i=1,#inputs do
    -- disp progress
    xlua.progress(i, #inputs)
    
    -- build test set
    local tmp = inputs[i][{{}, {2, (opt.input_dim+1)}}]
    
    local input, target = {}, {}
    table.insert(input, tmp)
    
    -- test sample
    local output = rnn:forward(input)    
    predictions[i] = output[1]:clone()
  end
  
  -- timing
  time = sys.clock() - time
  time = time / #inputs
  print("\n==> time to classify 1 sample = " .. (time*1000) .. 'ms')
    
  return predictions
end

dofile('utils.lua')

print('==> load test set')
data_dir = ''
if opt.data_type == 'dummy' then
  data_dir = '../data/dummy/'
elseif opt.data_type == 'vot' then
  data_dir = '../data/vot/'
end
local test_data = torch.load(paths.concat(data_dir, opt.test))
for i=1,#test_data do
  if test_data[i]:ne(test_data[i]):sum() > 0 then
      print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
      test_data[i][test_data[i]:ne(test_data[i])] = 0
  end
end

-- data statistics
print('\n==> data statistics: ')
print('==> number of test examples: ' .. #test_data)

print('\n==> loading model')
rnn = torch.load(opt.model_path)

-- evaluate on the test set
local y_hat_frame = classify(test_data)

-- extracting y
local y_frame = {}
for i=1,#test_data do
  local tmp = test_data[i][{{}, 1}]
  tmp:add(1)
  y_frame[i] = tmp
end

-- getting durations
y = calc_onset_offset(y_frame, true)
y_hat = calc_onset_offset(y_hat_frame, false)

-- plot stats
plot_classification_stats(y, y_hat)

-- write predictions to file
write_predictions(opt.dump_dir, y_hat, y_hat_frame)

--[[
diffs = {}
total_dif = 0
for i=1,#y do
  dif = torch.abs(y[i][1] - y_hat[i][1])
  print(string.format('Exmaple number: %d, Diff: %d', i, dif))
  if diffs[dif] == nil then
    diffs[dif] = 1
  else
    diffs[dif] = diffs[dif] + 1
  end
end

for k, v in pairs(diffs) do
  print(string.format('Difference: %dms, Accuracy percentage: %.2f%%', k, 100*(v/#y)))
end]]--
