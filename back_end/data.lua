--require('mobdebug').start()
require ('torch')
require ('nn')
require ('rnn')

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print('==> processing options')
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-Directional LSTM')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-data_type', 'neg_vot', 'the type of the data: lm | dummy | vot')
   cmd:option('-train', 'train.t7', 'the path to the train data')
   cmd:option('-test', 'test.t7', 'the path to the test data')
   cmd:option('-input_dim', 9, 'the input size')   
   cmd:option('-batch_size', 32, 'the mini-batch size')   
   cmd:option('-val_percentage', 0.2, 'the percentage of exampels to be considered as validation set from the training set')
   cmd:text()
   opt = cmd:parse(args or {})
end
----------------------------------------------------------------------

-- load from txt file
function load_data(path, num_of_examples, input_dim)
  local file = io.open(path, "r")
  local data = torch.Tensor(num_of_examples, input_dim)
  
  if file then
    local i = 1
    for line in file:lines() do      
      local j = 1
      for str in string.gmatch(line, "(%S+)") do
        data[i][j] = str
        j = j + 1
      end
      i = i + 1
    end
  else
    print("\n==>ERROR: can not open file.")    
  end  
  if not file then
    file:close()
  end
  return data
end

function get_mini_batch(x, y, rho)  
  mini_batch_idx = mini_batch_idx + 1
  start_idx = (mini_batch_idx-1)*rho + 1
  end_idx = (mini_batch_idx)*rho

  -- validation
  if x:size(1) < end_idx then 
    mini_batch_idx = 1
    start_idx = 1
    end_idx = rho
  end
  
  local mini_batch_x = x[{{start_idx, end_idx}, {}}]
  local mini_batch_y = y[{{start_idx, end_idx}}]
  
  return mini_batch_x, mini_batch_y
end

mini_batch_idx = 0 --for the minibatchs

print("==> Loading data set")
if opt.data_type == 'lm' then
  -- build dummy dataset (task is to predict next item, given previous)
  sequence_ = torch.LongTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
  sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
  sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

  offsets = {}
  for i=1,opt.batch_size do
     table.insert(offsets, math.ceil(math.random()*opt.batch_size))
  end
  offsets = torch.LongTensor(offsets)
  
elseif opt.data_type == 'dummy' then
  -- build dummy dataset, same as vowel duration
  data_dir = 'data/dummy/'
  local train = torch.load(paths.concat(data_dir, opt.train))
  local test = torch.load(paths.concat(data_dir, opt.test))
    
  train_x = train[{{}, {2, 10}}]
  train_y = train[{{}, 1}]
  train_y:add(1) -- torch start index is 1 not 0
  
  test_x = test[{{}, {2, 10}}]
  test_y = test[{{}, 1}]  
  test_y:add(1) -- torch start index is 1 not 0

elseif opt.data_type == 'neg_vot' then
  data_dir = 'data/neg_vot/fold_1/'
  local train = torch.load(paths.concat(data_dir, opt.train))
  local test = torch.load(paths.concat(data_dir, opt.test))
    
  -- take part of the training set for validation
  local val_size = train:size(1)*opt.val_percentage
  local val = train[{{(train:size(1)-val_size), train:size(1)}, {}}]   -- take the last elements for validation
  train = train[{{1, (train:size(1)-val_size-1)}, {}}]                 -- the rest are for training
  
  -- Detecting and removing NaNs
  if train:ne(train):sum() > 0 then
    print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
    train[train:ne(train)] = 0
  end
  if test:ne(test):sum() > 0 then
    print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
    test[test:ne(test)] = 0
  end
  if val:ne(val):sum() > 0 then
    print(sys.COLORS.red .. ' validation set has NaN/s, replace with zeros.')
    val[val:ne(val)] = 0
  end
  
  -- build training set
  train_x = train[{{}, {2, (opt.input_dim+1)}}]
  train_y = train[{{}, 1}]
  train_y:add(1) -- torch start index is 1 not 0

  -- build validation set
  val_x = val[{{}, {2, (opt.input_dim+1)}}]
  val_y = val[{{}, 1}]
  val_y:add(1) -- torch start index is 1 not 0

  -- build test set
  test_x = test[{{}, {2, (opt.input_dim+1)}}]
  test_y = test[{{}, 1}]  
  test_y:add(1) -- torch start index is 1 not 0    
  
elseif opt.data_type == 'multi' then
  data_dir = 'data/multi_class/'
  local train = torch.load(paths.concat(data_dir, opt.train))
  local test = torch.load(paths.concat(data_dir, opt.test))
    
  -- take part of the training set for validation
  local val_size = train:size(1)*opt.val_percentage
  local val = train[{{(train:size(1)-val_size), train:size(1)}, {}}]   -- take the last elements for validation
  train = train[{{1, (train:size(1)-val_size-1)}, {}}]                 -- the rest are for training
  
  -- Detecting and removing NaNs
  if train:ne(train):sum() > 0 then
    print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
    train[train:ne(train)] = 0
  end
  if test:ne(test):sum() > 0 then
    print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
    test[test:ne(test)] = 0
  end
  if val:ne(val):sum() > 0 then
    print(sys.COLORS.red .. ' validation set has NaN/s, replace with zeros.')
    val[val:ne(val)] = 0
  end
  
  -- build training set
  train_x = train[{{}, {2, (opt.input_dim+1)}}]
  train_y = train[{{}, 1}]
  train_y:add(1) -- torch start index is 1 not 0

  -- build validation set
  val_x = val[{{}, {2, (opt.input_dim+1)}}]
  val_y = val[{{}, 1}]
  val_y:add(1) -- torch start index is 1 not 0

  -- build test set
  test_x = test[{{}, {2, (opt.input_dim+1)}}]
  test_y = test[{{}, 1}]  
  test_y:add(1) -- torch start index is 1 not 0
  
elseif opt.data_type == 'vot' then
  data_dir = 'data/vot/'
  local train = torch.load(paths.concat(data_dir, opt.train))
  local test = torch.load(paths.concat(data_dir, opt.test))
    
  -- take part of the training set for validation
  local val_size = train:size(1)*opt.val_percentage
  local val = train[{{(train:size(1)-val_size), train:size(1)}, {}}]   -- take the last elements for validation
  train = train[{{1, (train:size(1)-val_size-1)}, {}}]                 -- the rest are for training
  
  -- Detecting and removing NaNs
  if train:ne(train):sum() > 0 then
    print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
    train[train:ne(train)] = 0
  end
  if test:ne(test):sum() > 0 then
    print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
    test[test:ne(test)] = 0
  end
  if val:ne(val):sum() > 0 then
    print(sys.COLORS.red .. ' validation set has NaN/s, replace with zeros.')
    val[val:ne(val)] = 0
  end
  
  -- build training set
  train_x = train[{{}, {2, (opt.input_dim+1)}}]
  train_y = train[{{}, 1}]
  train_y:add(1) -- torch start index is 1 not 0

  -- build validation set
  val_x = val[{{}, {2, (opt.input_dim+1)}}]
  val_y = val[{{}, 1}]
  val_y:add(1) -- torch start index is 1 not 0

  -- build test set
  test_x = test[{{}, {2, (opt.input_dim+1)}}]
  test_y = test[{{}, 1}]  
  test_y:add(1) -- torch start index is 1 not 0    
else
  print("\nNo such data set")
end