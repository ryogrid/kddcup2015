# -*- coding: utf-8 -*-
require 'torch'
require 'nn'

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

model = nn.Sequential();
model:add(nn.Linear(8,2000))
model:add(nn.Tanh())
model:add(nn.Linear(2000, 2000))
model:add(nn.Tanh())
model:add(nn.Linear(2000, 1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01 --学習係数
trainer.maxIteration = 100  --学習回数
trainer:train(train_dataset)

model:evaluate()

ft = io.open("../test/torch_input_test.csv", "r")
fr = io.open("../test/resut_simple_nn_0621_2.csv", "w")
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
    fr:write(input_data[1] .. "," .. model:forward(x)[1] .. "\n")
end
