-- for debuing purposes
require('mobdebug').start()
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

   cmd:option('-data_type', 'neg_vot', 'the type of the data: dummy | vot | neg_vot')
   cmd:option('-test', 'classification_test.t7', 'the path to the test data')
   cmd:option('-input_dim', 63, 'the input size')
   cmd:option('-model_path', '../results/neg_vot_measurement_dimitrieva/model.net', 'the path to the model') 
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')
  
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

function predict_classes(frame_pred)
  local classes = {}
  for i=1,#frame_pred do
    maxs, dims = torch.max(frame_pred[i], 2)
    classes[i] = dims
  end  
  return classes
end

function calc_onset_offset(frame_pred, is_target)
  local onset_offset = {}
  
  if not is_target then
    local classes = predict_classes(frame_pred)  
    for i=1, #classes do
      local onset = 0
      local offset = 0
      for j=2,classes[i]:size(1) do
        if classes[i][j-1][1] == 1 and classes[i][j][1] == 2 then
          onset = j - 1
        elseif classes[i][j-1][1] == 2 and classes[i][j][1] == 1 then
          offset = j - 1
        end
      end
      local curr_y = {}
      table.insert(curr_y, onset)
      table.insert(curr_y, offset)
      onset_offset[i] = curr_y
    end
  else
    classes = frame_pred
    for i=1, #classes do
      local onset = 0
      local offset = 0
      for j=2,classes[i]:size(1) do
        if classes[i][j-1] == 1 and classes[i][j] == 2 then
          onset = j
        elseif classes[i][j-1] == 2 and classes[i][j] == 1 then
          offset = j - 1
        end
      end
      local curr_y = {}
      table.insert(curr_y, onset)
      table.insert(curr_y, offset)
      onset_offset[i] = curr_y
    end
  end
  return onset_offset
end

dofile('utils.lua')

print('==> load test set')
data_dir = ''
if opt.data_type == 'dummy' then
  data_dir = '../data/dummy/'
elseif opt.data_type == 'vot' then
  data_dir = '../data/vot/'
elseif opt.data_type == 'neg_vot' then
  data_dir = '../data/neg_vot/'
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

y = calc_onset_offset(y_frame, true)
y_hat = calc_onset_offset(y_hat_frame, false)

task_loss = 0
ms_2 = 0
ms_5 = 0
ms_10 = 0
ms_15 = 0
ms_25 = 0
ms_50 = 0

for i=1,#y do
  curr_y = y[i]
  curr_y_hat = y_hat[i]  
  task_loss = task_loss + torch.abs(curr_y[1] - curr_y_hat[1])
  
  y_dur = curr_y[1]
  y_hat_dur = curr_y_hat[1]
  dif = torch.abs(y_dur - y_hat_dur)
  
  if dif <= 2 then
    ms_2 = ms_2 + 1
  end
  if dif <= 5 then
    ms_5 = ms_5 + 1
  end
  if dif <= 10 then
    ms_10 = ms_10 + 1
  end
  if dif <= 15 then
    ms_15 = ms_15 + 1
  end
  if dif <= 25 then
    ms_25 = ms_25 + 1
  end
  if dif <= 50 then
    ms_50 = ms_50 + 1
  end
end

task_loss = task_loss/#y
ms_2 = ms_2/#y
ms_5 = ms_5/#y
ms_10 = ms_10/#y
ms_15 = ms_15/#y
ms_25 = ms_25/#y
ms_50 = ms_50/#y

print('\n==> cumulative task loss: ' .. task_loss)
print('\n==> <= 2ms: ' .. ms_2*100 .. '%')
print('==> <= 5ms: ' .. ms_5*100 .. '%')
print('==> <= 10ms: ' .. ms_10*100 .. '%')
print('==> <= 15ms: ' .. ms_15*100 .. '%')
print('==> <= 25ms: ' .. ms_25*100 .. '%')
print('==> <= 50ms: ' .. ms_50*100 .. '%')
