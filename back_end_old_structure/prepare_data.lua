--require('mobdebug').start()
lfs = require 'lfs'
require 'xlua'    -- xlua provides useful tools, like progress bars

-- load the onset and offset times
-- we assume there is no header to this file!!
function load_labels(path, path_names)
  local file = io.open(path, "r")
  local labels = {}
  -- validate the file descriptor
  if file then
    -- loop over all the labels
    for line in file:lines() do
      -- parse the onset and offsets
      onset_offset = {}
      for str in string.gmatch(line, "(%S+)") do
        table.insert(onset_offset, str)
      end
      table.insert(labels, onset_offset)
    end  
    if file then
      file:close()
    end
  else
    print("\n==> Error: cannot read file.")
  end
  
  file = io.open(path_names, "r")
  local names = {}
  -- validate the file descriptor
  if file then
    -- loop over all the labels
    for line in file:lines() do
      -- parse the onset and offsets
      table.insert(names, line)
    end  
    if file then
      file:close()
    end
  else
    print("\n==> Error: cannot read file.")
  end
  
  -- create a hash table between the filename and its label
  data = {}
  for i=1,#labels do
    data[names[i]] = labels[i]
  end
  return data
end

-- load from txt file
function load_data(path)
  local file = io.open(path, "r")  
  local data = ""
  -- validate the file descriptor
  if file then
    local i = 1
    local head = true
    -- loop over all the features
    for line in file:lines() do
      -- parse the header
      if head then
        dims = {}
        for str in string.gmatch(line, "(%S+)") do
          table.insert(dims, str)
        end
        -- get the current example dimensions
        data = torch.Tensor(tonumber(dims[1]), tonumber(dims[2]))
        head = false
      else
      -- parse the features
        local j = 1      
        for str in string.gmatch(line, "(%S+)") do
          data[i][j] = str
          j = j + 1
        end
        i = i + 1
      end
    end
  else
    print("\n==> ERROR: cannot read file.")
  end
  if file then
    file:close()
  end
  return data, tonumber(dims[1])
end

-- creating the dataset
function build_test_dataset(path, labels_filename, names_filename)    
  print('==> compute data statistics')
  -- for debuging purposes
  -- counting the files
  local t = 0
  for file in lfs.dir(path) do
    -- disp progress
    -- get all the .txt files
    if string.sub(file, #file-3, -1) == ".txt" then 
      t = t + 1
    end
  end
  
  print('==> loading labels')
  -- load all the labels onset and offsets
  labels = load_labels(paths.concat(path, labels_filename), paths.concat(path, names_filename))
  
  print('==> loading features')
  local all_data = {}
  local total = t
  t = 0
  -- loop over all the files in path
  for file in lfs.dir(path) do
    -- disp progress
    -- get all the .txt files
    if string.sub(file, #file-3, -1) == ".txt" then 
      -- progress bar
      t = t + 1
      xlua.progress(t, total)
      
      --load the data and its dim
      local x, dim = load_data(paths.concat(path, file))
      local y = torch.zeros(dim, 1)
      y[{{labels[file][1], labels[file][2]}, 1}] = 1
      local x_y = torch.zeros(x:size(1), x:size(2)+1)
      x_y[{{}, 1}] = y
      x_y[{{}, {2, x_y:size(2)}}] = x

      -- validation
      if dim < tonumber(labels[file][2]) then 
        print('ERROR')
      end
      
      -- aggregate all the features and labels
      table.insert(all_data, x_y)  
    end
  end
  
  return all_data
end

-- creating the dataset
function build_dataset(path, labels_filename, names_filename)    
  local num_of_examples = 0
  local x_size = 0
  
  print('==> compute data statistics')
  -- get statistics of the data  
  -- compute the size total size of examples and features
  -- loop over all the files in path
  local t = 0
  for file in lfs.dir(path) do
    -- disp progress
    -- get all the .txt files
    if string.sub(file, #file-3, -1) == ".txt" then 
      t = t + 1
      local file = io.open(paths.concat(path, file), "r")
      local header = false -- flag to indicate the if we already processed the header of each file
      for line in file:lines() do  
        -- process only the header 
        if not header then
          local flag = 1
          -- append the dimensions of the current file
          for str in string.gmatch(line, "(%S+)") do
            if flag == 1 then
              num_of_examples = num_of_examples + str
            else
              x_size = str
            end
            flag = flag + 1
          end
          header = true
        else
          break
        end        
      end
      if file then
          file:close()
      end
    end
  end
  
  -- create the data matrix for all the files and labels together
  local x = torch.zeros(num_of_examples, x_size)
  local y = torch.zeros(num_of_examples, 1)
  local x_count = 1
  local y_count = 1
  
  print('==> loading labels')
  -- load all the labels onset and offsets
  labels = load_labels(paths.concat(path, labels_filename), paths.concat(path, names_filename))
  
  local idx = 1
  local total = t
  t = 0
  print('==> loading features')
  -- load the features
  -- loop over all the files in path  
  for file in lfs.dir(path) do    
    -- process only .txt files
    if string.sub(file, #file-3, -1) == ".txt" then  
      -- progress bar
      xlua.progress(t, total)
      t = t + 1
      
      --load the data and its dim
      local tmp, dim = load_data(paths.concat(path, file))
      -- append the current dim to the total dim
      local tmp_x_count = x_count + dim
      
      -- insert the current example into the total data and labels
      x[{{x_count, tmp_x_count-1}, {}}] = tmp
      y[{{y_count+labels[file][1], y_count+labels[file][2]}, 1}] = 1
      
      if dim < tonumber(labels[file][2]) then 
        print('ERROR')
      end
      
      x_count = tmp_x_count
      y_count = y_count+ dim
      idx = idx + 1
    end
  end
  -- merge the features and targets to the same matrix
  -- the targets will be at the first column and the features at the rest of the matrix
  local data = torch.zeros(x:size(1), x:size(2)+1)
  data[{{}, 1}] = y
  data[{{}, {2, data:size(2)}}] = x
  return data
end

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Prepare dataset for the Bi-Directional LSTM for VOT measurement.')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-input_dir', '/Users/yossiadi/Datasets/vot/natalia/features/prevoiced/', 'the path to the dir where the data files are placed (in a .txt format)')   
   cmd:option('-output_file', 'data/vot/prevoiced_natalia_test.t7', 'the path to save the t7 binary file')   
   cmd:option('-labels_filename', 'labels.dat', 'the path to the labels filename')
   cmd:option('-names_filename', 'names.dat', 'the path to the labels filename')
   cmd:option('-is_test', true, 'boolean flag which indicates of we build the train or test dataset')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

if opt.is_test == false then
  -- create the dataset
  data = build_dataset(opt.input_dir, opt.labels_filename, opt.names_filename)
  print('\n==> saving data to ' .. (opt.output_file))
  torch.save(opt.output_file, data)
  print('\n==> Done.')
else
  -- create the dataset
  data = build_test_dataset(opt.input_dir, opt.labels_filename, opt.names_filename)
  print('\n==> saving data to ' .. (opt.output_file))
  torch.save(opt.output_file, data)
  print('\n==> Done.')
end