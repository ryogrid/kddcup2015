# -*- coding: utf-8 -*-
require 'torch'
require 'nn'

dataset={};
function dataset:size() return 100 end
for i=1,dataset:size() do
   local input = torch.randn(2);
   local output = torch.Tensor(1);
   if input[1]*input[2]>0 then
      output[1] = -1;
   else
      output[1] = 1
   end
   dataset[i] = {input, output}
end

model = nn.Sequential();
model:add(nn.Linear(2,50))
model:add(nn.Tanh())
model:add(nn.Linear(50, 1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01 --学習係数
trainer.maxIteration = 100  --学習回数
trainer:train(dataset)

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(model:forward(x))
x[1] =  0.5; x[2] = -0.5; print(model:forward(x))
x[1] = -0.5; x[2] =  0.5; print(model:forward(x))
x[1] = -0.5; x[2] = -0.5; print(model:forward(x))
