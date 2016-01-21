require 'nn'

net = nn.Sequential()
net:add(nn.Linear(1, 1))

local function print_net_params(net)
weights, gradWeights = net:parameters()
    print('START')
    print('weight = ')
    print(weights[1][1][1])
    print('bias = ')
    print(weights[2][1])
    print('gradWeight = ')
    print(gradWeights[1][1][1])
    print('gradBias = ')
    print(gradWeights[2][1])
    print('END')
end

print_net_params(net)

input = torch.Tensor{3}
print('input = ')
print(input[1])

output = net:forward(input)
print('output = ')
print(output[1])

gradOut = torch.Tensor{1}
print('gradOut = ')
print(gradOut[1])

net:zeroGradParameters()

---gradInput = net:backward(input, gradOut)
---print('gradInput = ')
---print(gradInput[1])

print_net_params(net)
net:accGradParameters(input, gradOut)
net:updateParameters(1)
print_net_params(net)
