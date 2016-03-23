-- for debuing purposes
--require('mobdebug').start()
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
   cmd:text('Negative VOT prediction')
   cmd:text()
   cmd:text('Options:')
   
   cmd:option('-threads', 4, 'number of threads')
   cmd:option('-type', 'double', 'the data type: double')
   cmd:option('-model_path', 'results/model.net', 'the path to the saved model')
   -- data:
   cmd:option('-input_dim', 900, 'The input size') 
   
   -- loss:
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | margin')
   
   cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
   
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
torch.setnumthreads(opt.threads)

model = torch.load(opt.model_path)

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '5_test.lua'
----------------------------------------------------------------------

-- evaluating of test set
-- local vars
local time = sys.clock()

-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
model:evaluate()

-- test over test data
print('==> testing on test set:')
for t = 1,testData:size() do
  -- disp progress
  xlua.progress(t, testData:size())

  -- get new sample
  local input = testData.data[t]
  if opt.type == 'double' then input = input:double()
  elseif opt.type == 'cuda' then input = input:cuda() end
  local target = testData.labels[t]

  -- test sample
  local pred = model:forward(input)
  confusion:add(pred, target)
end

-- timing
time = sys.clock() - time
time = time / testData:size()
print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
print(confusion)
precision = confusion.mat[1][1] / (confusion.mat[1][1] + confusion.mat[1][2])
recall = confusion.mat[1][1] / (confusion.mat[1][1] + confusion.mat[2][1])
f_measure = 2 * (precision * recall) /  (precision + recall)
acc = (confusion.mat[1][1] + confusion.mat[2][2]) / (confusion.mat[1][1] + confusion.mat[1][2] + confusion.mat[2][1] + confusion.mat[2][2])

print('Accuracy: ' .. acc)
print('Precision: ' .. precision)
print('Recall: ' .. recall)
print('F1-Measure: ' .. f_measure)
print('==> Done.')

