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
   cmd:option('-data_type', 'vot', 'the type of the data: multi | dummy | vot')
   cmd:option('-train', 'train.t7', 'the path to the train data')
   cmd:option('-test', 'test.t7', 'the path to the test data')
   cmd:option('-val', 'val.t7', 'the path to the test data')
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


function pop_data(orig_data)
  local data = {}
  for i=1, #orig_data[1] do
    local y = torch.zeros(orig_data[1][i]:size(1))
    y[{{tonumber(orig_data[2][i][1]), tonumber(orig_data[2][i][2])}}] = 1
    local item = torch.zeros(orig_data[1][i]:size(1), orig_data[1][i]:size(2), orig_data[1][i]:size(3)+1)
    item[{{}, {}, 1}] = y
    item[{{}, {}, {2, item:size(3)}}] = orig_data[1][i]

    table.insert(data, item)
  end
  return data
end


mini_batch_idx = 0 --for the minibatchs
print("==> Loading data set")
if opt.data_type == 'dummy' then
  -- build dummy dataset, same as vowel duration
  data_dir = 'data/dummy/'
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
  all_data = torch.load(paths.concat(data_dir, opt.train))
  all_test_data = torch.load(paths.concat(data_dir, opt.test))
    
  -- take part of the training set for validation
  local val_size = #all_data*opt.val_percentage  
  local train_size = #all_data - val_size
  val_data = {}
  train_data = {}
  for i=1,#all_data do
    -- Detecting and removing NaNs
    if all_data[i]:ne(all_data[i]):sum() > 0 then
      print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
      all_data[i][all_data[i]:ne(all_data[i])] = 0
    end
    if i <= train_size then
      table.insert(train_data, all_data[i]:reshape(all_data[i]:size(1), 1, all_data[i]:size(2)))
    else
      table.insert(val_data, all_data[i]:reshape(all_data[i]:size(1), 1, all_data[i]:size(2)))
    end
  end
  
  test_data = {}
  for i=1,#all_test_data do
    -- Detecting and removing NaNs
    if all_test_data[i]:ne(all_test_data[i]):sum() > 0 then
      print(sys.COLORS.red .. ' test set has NaN/s, replace with zeros.')
      all_test_data[i][all_test_data[i]:ne(all_test_data[i])] = 0
    end
    table.insert(test_data, all_test_data[i]:reshape(all_test_data[i]:size(1), 1, all_test_data[i]:size(2)))
  end
  
elseif opt.data_type == 'vot' then
  data_dir = 'data/bb_pos/'
  local train = torch.load(paths.concat(data_dir, opt.train))
  local test = torch.load(paths.concat(data_dir, opt.test))
  local val = torch.load(paths.concat(data_dir, opt.val))
  
  train_data = pop_data(train)
  val_data = pop_data(val)
  test_data = pop_data(test)
else
  print("\nNo such data set")
end