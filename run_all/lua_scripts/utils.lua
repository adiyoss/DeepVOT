--require('mobdebug').start()

function predict_classes(frame_pred)
  local classes = {}
  for i=1,#frame_pred do
    maxs, dims = torch.max(frame_pred[i], 2)
    classes[i] = dims
  end  
  return classes
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

function calc_onset_offset(frame_pred, is_target)
  local onset_offset = {}
  if not is_target then
    local classes = predict_classes(frame_pred)  
    for i=1, #classes do
      local best_onset = 0
      local best_offset = 0
      local best_num_of_frams = 0
      local onset = 0
      local offset = 0
      for j=2,classes[i]:size(1) do
        if classes[i][j-1][1] == 1 and classes[i][j][1] == 2 then
          onset = j - 1
        elseif classes[i][j-1][1] == 2 and classes[i][j][1] == 1 then
          offset = j - 1
          local tmp = offset - onset
          if best_num_of_frams < tmp then
            best_num_of_frams = tmp
            best_onset = onset
            best_offset = offset
          end
        end
      end
      local curr_y = {}
      table.insert(curr_y, best_onset)
      table.insert(curr_y, best_offset)
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

function calc_onset_offset_neg(frame_pred, is_target)
  local onset_offset = {}
  
  if not is_target then
    local classes = predict_classes(frame_pred)  
    for i=1, #classes do
      local onset = 0
      local best_onset = 0
      local tmp_num_of_frames = 0
      local num_of_frames = 0
    
      for j=2,classes[i]:size(1) do
        if (j == classes[i]:size(1) - 1) then
          if num_of_frames < tmp_num_of_frames then
              best_onset = onset              
              num_of_frames = tmp_num_of_frames
              tmp_num_of_frames = 0
              break
          end
        end        
        if classes[i][j-1][1] == 1 and classes[i][j][1] == 2 then
          onset = j - 1
          tmp_num_of_frames = 1
        elseif classes[i][j-1][1] == 2 and classes[i][j][1] == 2 then
          tmp_num_of_frames = tmp_num_of_frames + 1
        elseif (classes[i][j-1][1] == 2 and classes[i][j][1] == 1) then
          if num_of_frames < tmp_num_of_frames then
              best_onset = onset              
              num_of_frames = tmp_num_of_frames
              tmp_num_of_frames = 0
          end
        end
      end
      local curr_y = {}
      table.insert(curr_y, best_onset)
      table.insert(curr_y, 0)
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


function plot_classification_stats_neg_vot(y, y_hat)
  -- statistics
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
end

function plot_classification_stats(y, y_hat)
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
    task_loss = task_loss + torch.abs(curr_y[1] - curr_y_hat[1]) + torch.abs(curr_y[2] - curr_y_hat[2])
    
    y_dur = curr_y[2] - curr_y[1]
    y_hat_dur = curr_y_hat[2] - curr_y_hat[1]
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
end

function write_predictions(dump_dir, y_hat, y_hat_frame)
  os.execute('mkdir -p ' .. dump_dir)
  
  -- write the predctions file
  print('==> write onset predictions')
  out_fid = io.open(paths.concat(dump_dir, 'durations.txt'), 'w')
  for i=1,#y_hat do
    out_fid:write(tostring(y_hat[i][1]) .. ' ' .. tostring(y_hat[i][2]) .. '\n')
  end
  out_fid:close()
  
  -- write the predctions file
  print('==> write full predictions')
  local classes = predict_classes(y_hat_frame) 
  out_fid = io.open(paths.concat(dump_dir, 'full_predictions.txt'), 'w')
  for i=1,#classes do
    for j=1,classes[i]:size(1) do
      out_fid:write(tostring(classes[i][j][1]) .. ' ')
    end
    out_fid:write('\n')
  end
  out_fid:close()
end

function write_raw_predictions(output_filename, y_hat_frame)
  -- write the predctions file
  print('==> write full predictions')
  out_fid = io.open(output_filename, 'w')
  for j=1,y_hat_frame:size(1) do
    out_fid:write(tostring(y_hat_frame[j][1]) .. ' ' .. tostring(y_hat_frame[j][2]) .. ' ' .. tostring(y_hat_frame[j][3]) .. ' ' .. tostring(y_hat_frame[j][4]) .. '\n')
  end
  out_fid:close()
end