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
   cmd:option('-loss', 'nll', 'the type of loss function: nll')      
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------

if opt.loss == 'nll' then
  -- build criterion
  criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
else
  print("\nERROR: such loss function is not supported, set default loss to negative log likelihood")
  criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
end

print(criterion)