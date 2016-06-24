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

   cmd:option('-input_file', '../test_data/features/f.txt', 'the path to the features file')   
   cmd:option('-output_file', '1.txt', 'the path to output the predictions')
   cmd:option('-input_dim', 63, 'the input size')
   cmd:option('-model_path', 'model/deep_vot_model.net', 'the path to the model')
   cmd:option('-output_dim', 4, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden layer size')

   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

-- classify function
function classify(x) 
  -- local vars
  local time = sys.clock()
  local prediction
  
  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  rnn:evaluate()

  -- predict over test data
  print('==> classifing:')     
  local input = {}
  for t=1, x:size(1) do
    table.insert(input, x[t])
  end
      
  -- test sample
  local output = rnn:forward(input)    
--  prediction = output[1]:clone()

  -- timing
  time = sys.clock() - time
  print("\n==> time to classify 1 sample = " .. (time*1000) .. 'ms')
    
  return output
end

------------------------------------------------------------------
dofile('utils.lua')

print('==> load test file')
local test_data = load_data(opt.input_file)

if test_data:ne(test_data):sum() > 0 then
    print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
    test_data[test_data:ne(test_data)] = 0
end

-- data statistics
print('\n==> loading model')
rnn = torch.load(opt.model_path)

-- evaluate on the test set
local y_hat_frame = classify(test_data)

-- write predictions to file
write_raw_predictions(opt.output_file, y_hat_frame)
