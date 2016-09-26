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
   cmd:option('-input_dim', 9, 'the input size')   
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-hidden_size', 100, 'the hidden size')
   cmd:option('-dropout', 0.3, 'dropout rate')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

rnn = nn.Sequential()
rnn:add(nn.SeqLSTM(opt.input_dim, opt.hidden_size))
rnn:add(nn.Sequencer(nn.Dropout(opt.dropout)))
rnn:add(nn.SeqLSTM(opt.hidden_size, opt.hidden_size))
rnn:add(nn.Sequencer(nn.Dropout(opt.dropout)))
rnn:add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.output_dim))) -- times two due to JoinTable
rnn:add(nn.Sequencer(nn.LogSoftMax()))
  
-- print the model
print(rnn)