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
function build_neg_dataset(path, labels_filename, names_filename, num_feat)    
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
      new_x = x[{{}, {1, num_feat}}]:reshape(x:size(1)*num_feat)
      new_x = new_x:resize(10 * num_feat) -- TODO!!!!!
      local y = labels[file][1]
      local x_y = torch.zeros(new_x:size(1) + 1, 1)
      x_y[{1,1}] = y
      x_y[{{2, x_y:size(1)}, {}}] = new_x
      
      -- aggregate all the features and labels
      table.insert(all_data, x_y)  
    end
  end
  
  return all_data
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
   cmd:option('-input_dir', '/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/10ms_features/fold_1/train/', 'the path to the dir where the data files are placed (in a .txt format)')      
   cmd:option('-output_file', 'data/10ms_db/fold_1/train.t7', 'the path to save the t7 binary file')   
   cmd:option('-labels_filename', 'labels.dat', 'the path to the labels filename')
   cmd:option('-names_filename', 'names.dat', 'the path to the labels filename')
   cmd:option('-num_features', 9, 'the number of features to use')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

-- create the dataset
data = build_neg_dataset(opt.input_dir, opt.labels_filename, opt.names_filename, opt.num_features)
print('\n==> saving data to ' .. (opt.output_file))
torch.save(opt.output_file, data)
print('\n==> Done.')
