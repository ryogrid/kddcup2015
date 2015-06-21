# -*- coding: utf-8 -*-
require 'torch'
require 'nn'
require 'unsup'
require 'image'
require 'optim'
--require 'autoencoder-data'

-- split関数
function split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end

    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local lastPos
    for part, pos in string.gfind(str, pat) do
        table.insert(result, part)
        lastPos = pos
    end
    table.insert(result, string.sub(str, lastPos))
    return result
end
					    
train_dataset={}

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'linear', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 16, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', true, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 10000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 9, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile', 'http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', false, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:addTime('psd')
cmd:log(params.rundir .. '/log.txt', params)

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)

-- io.openで、ファイルを開く
fi = io.open("./torch_input.csv", "r")
fo = io.open("./truth_train.csv", "r")

data_num = 1
output_arr={}
for oline in fo:lines() do
     local output_data = split(oline, ",")
     output_arr[data_num] = tonumber(output_data[2])
     data_num = data_num + 1
end

data_num = 1
-- f:linesで一行ずつテキストファイルを読み込む
for iline in fi:lines() do
   local input = torch.Tensor(8)
   local input_data = split(iline, ",")
   input[1] = tonumber(input_data[2])
   input[2] = tonumber(input_data[3])
   input[3] = tonumber(input_data[4])
   input[4] = tonumber(input_data[5])
   input[5] = tonumber(input_data[6])
   input[6] = tonumber(input_data[7])
   input[7] = tonumber(input_data[8])
   input[8] = tonumber(input_data[9])  

   local output = torch.Tensor(1)
   output[1] = output_arr[data_num]
   
   train_dataset[data_num] = {input, output}
   data_num = data_num + 1
end

function train_dataset:size() return 120542 end

-- encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(8,1))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(8))

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(1,8))

-- complete model
module = unsup.AutoEncoder(encoder, decoder, 1)

-- are we using the hessian?
if params.hessian then
   nn.hessian.enable()
   module:initDiagHessianParameters()
end

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()

--criterion = nn.MSECriterion()
--trainer = nn.StochasticGradient(model, criterion)
--trainer.learningRate = 0.01 --学習係数
--trainer.maxIteration = 1  --学習回数
--trainer:train(train_dataset)
--model:evaluate()


print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do
   -- update diagonal hessian parameters
   --
   if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
       -- some extra vars:
       local hessiansamples = params.hessiansamples
       local minhessian = params.minhessian
       local maxhessian = params.maxhessian
       local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
       etas = etas or ddl_ddx:clone()

       print('==> estimating diagonal hessian elements')
       for i = 1,hessiansamples do
             -- next
             local ex = train_dataset[i]
             local input = ex[1]
             local target = ex[2]
             module:updateOutput(input, target)

             -- gradient
	     dl_dx:zero()
	     module:updateGradInput(input, target)
	     module:accGradParameters(input, target)

	     -- hessian
	     ddl_ddx:zero()
	     module:updateDiagHessianInput(input, target)
	     module:accDiagHessianParameters(input, target)

             -- accumulate
	     ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
       end

       -- cap hessian params
       print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
       ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
       ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
       print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

       -- generate learning rates
       etas:fill(1):cdiv(ddl_ddx_avg)
   end

   -- progress
   --
   iter = iter+1
   xlua.progress(iter, params.statinterval)

   -- create mini-batch
   --
   local example = dataset[t]
   local inputs = {}
   local targets = {}
   for i = t,t+params.batchsize-1 do
      -- load new sample
      local sample = train_dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      table.insert(inputs, input)
      table.insert(targets, target)
   end

   -- define eval closure
   --
   local feval = function()
       -- reset gradient/f
       local f = 0
       dl_dx:zero()

       -- estimate f and gradients, for minibatch
       for i = 1,#inputs do
       -- f
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
       end

       -- normalize
       dl_dx:div(#inputs)
       f = f/#inputs

       -- return f and df/dx
       return f,dl_dx
   end

   -- compute statistics / report error
   --
   if math.fmod(t , params.statinterval) == 0 then
      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)
      err = 0; iter = 0
   end
end

ft = io.open("../test/torch_input_test.csv", "r")
fr = io.open("../test/resut_simple_nn_0621_deep1.csv", "w")
for tiline in ft:lines() do
    local x = torch.Tensor(8)
    local input_data = split(tiline, ",")
    x[1] = tonumber(input_data[2])
    x[2] = tonumber(input_data[3])
    x[3] = tonumber(input_data[4])
    x[4] = tonumber(input_data[5])
    x[5] = tonumber(input_data[6])
    x[6] = tonumber(input_data[7])
    x[7] = tonumber(input_data[8])
    x[8] = tonumber(input_data[9])
--      print(input_data[1])
--    print(model:forward(x)[1])    
    fr:write(input_data[1] .. "," .. module:forward(x)[1] .. "\n")
end
