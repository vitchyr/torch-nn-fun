-- See http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4
require 'dpnn'

function trainEpoch(module, criterion, inputs, target)
    for i = 1, inputs:size(1) do
        local indx = math.random(1, inputs:size(1))
        local input, target = inputs[idx], targets:narrow(1, idx, 1)
        -- forward
        local output = module:forward(input)
        local loss = criterion:forward(output, target)
        -- backward
        local gradOutput = criterion:backward(output, target)
        module:zeroGradParameters()
        local gradInput = module:backward(input, gradOutput)
        -- update
        module:updateGradParameters(0.9) -- momentum (dpnn)
        module:updateParameters(0.1) -- W = W - 0.1 dL/dW
    end
end

for i = 1, 100 do
    trainEpoch(module, criterion, inputs, targets)
end
