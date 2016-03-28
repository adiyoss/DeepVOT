require('nn')
require('rnn')

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Bi-Directional LSTM')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model_type', 'bi-lstm', 'the type of model: bi-lm | bi-lstm | lstm')   
   cmd:option('-input_dim', 9, 'the input size')   
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden size')
   cmd:option('-dropout', 0.3, 'dropout rate')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

if opt.model_type == 'bi-lm' then  
  -- forward rnn
  -- build simple recurrent neural network
  fwd = nn.Recurrent(
     opt.hidden_size, nn.LookupTable(opt.input_dim, opt.hidden_size), 
     nn.Linear(opt.hidden_size, opt.hidden_size), nn.Tanh(), 
     rho
  )

  -- backward rnn (will be applied in reverse order of input sequence)
  bwd = fwd:clone()
  bwd:reset() -- reinitializes parameters

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  merge = nn.JoinTable(1, 1)
  
  -- we use BiSequencerLM because this is a language model (previous and next words to predict current word).
  -- If we used BiSequencer, x[t] would be used to predict y[t] = x[t] (which is cheating).
  -- Note that bwd and merge argument are optional and will default to the above.
  brnn = nn.BiSequencerLM(fwd, bwd, merge)
  rnn = nn.Sequential()
     :add(brnn) 
     :add(nn.Sequencer(nn.Linear(2*opt.hidden_size, opt.output_dim))) -- times two due to JoinTable
     :add(nn.Sequencer(nn.LogSoftMax()))
     
elseif opt.model_type == 'bi-lstm' then 
  -- forward rnn
  -- build LSTM based rnn
  fwd = nn.FastLSTM(opt.input_dim, opt.hidden_size)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwd = fwd:clone()
  bwd:reset() -- reinitializes parameters

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  merge = nn.JoinTable(1, 1)
  -- merge = nn.CSubTable()
  
  -- build the bidirectional lstm
  brnn = nn.BiSequencer(fwd, bwd, merge)

  rnn = nn.Sequential()
     :add(brnn) 
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(2*opt.hidden_size, opt.output_dim))) -- times two due to JoinTable
     --:add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.output_dim))) -- 1 time due to CSubTable
     :add(nn.Sequencer(nn.LogSoftMax()))
     
elseif opt.model_type == 'lstm' then 
  
  --- SINGLE LAYER ---
  -- forward rnn
  -- build simple recurrent neural network
  --[[fwd_1 = nn.FastLSTM(opt.input_dim, opt.hidden_size)  
  s_rnn_1 = nn.Sequencer(fwd_1)

  rnn = nn.Sequential()     
     :add(s_rnn_1)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.output_dim)))
     :add(nn.Sequencer(nn.LogSoftMax()))]]--
     
  --- TWO LAYERS ---
  -- forward rnn
  -- build simple recurrent neural network
  fwd_1 = nn.FastLSTM(opt.input_dim, opt.hidden_size)  
  fwd_2 = nn.FastLSTM(opt.hidden_size, opt.hidden_size)  
  s_rnn_1 = nn.Sequencer(fwd_1)
  s_rnn_2 = nn.Sequencer(fwd_2)

  rnn = nn.Sequential()     
     :add(s_rnn_1)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(s_rnn_2)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.output_dim)))
     :add(nn.Sequencer(nn.LogSoftMax()))    
end

-- print the model
print(rnn)