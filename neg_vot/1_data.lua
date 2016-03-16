----------------------------------------------------------------------
----------------------------------------------------------------------
--require('mobdebug').start()
require 'torch'   -- torch
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Building db')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-input_dim', 900, 'The input size') 
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> Building db'
in_dim_size = opt.input_dim

db_dir = 'data/neg_vot/'
train_file = 'train.t7'
test_file = 'test.t7'

db = torch.load(paths.concat(db_dir, train_file))
y_train = torch.zeros(#db)
x_train = torch.zeros(#db, in_dim_size)
for i=1,#db do
  y_train[i] = db[i][1] + 1 -- torch start index is 1
  x_train[i] = db[i][{{2, in_dim_size+1}, {}}]
end

db = torch.load(paths.concat(db_dir, test_file))
y_test = torch.zeros(#db)
x_test = torch.zeros(#db, in_dim_size)
for i=1,#db do
  y_test[i] = db[i][1] + 1 -- torch start index is 1
  x_test[i] = db[i][{{2, in_dim_size+1}, {}}]
end

--[[x_train = torch.ones(100, 9)
y_train = torch.ones(100)

x_test = torch.ones(10, 9)
y_test = torch.ones(10)

x_train[{{1, 50}, {}}] = -1
y_train[{{1, 50}}] = 2

x_test[{{1, 5}, {}}] = -1
y_test[{{1, 5}}] = 2
]]--

trsize = x_train:size(1)
tesize = x_test:size(1)


trainData = {
   data = x_train,
   labels = y_train,
   size = function() return trsize end
}

testData = {
   data = x_test,
   labels = y_test,
   size = function() return tesize end
}