-- See http://on-demand.gputechconf.com/gtc/2015/webinar/torch7-applied-deep-learning-for-vision-natural-language.mp4
local nn = require 'nn'
local dp = require 'dp'

local cnn = nn.Sequential()
cnn:add(nn.Convert('bhwc', 'bchw')) -- cast input to same type as cnn
-- 2 conv layers
cnn:add(nn.SpatialConvolution(1, 16, 5, 5, 1, 1, 2, 2))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- 1 dense hidden layer
local outsize = cnn:outside{1, 28, 28, 1} -- output size of convolutions
cnn:add(nn.Collapse(3))
cnn:add(nn.Linear(outsize[2]*outsize[3]*outsize[4], 200))
cnn:add(nn.ReLU())
-- output layer
cnn:add(nn.Linear(200, 10))
cnn:add(nn.LogSoftMax())
print(cnn)

train = dp.Optimizer{
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
    callback = function(model, report)
        model:updateGradParameters(0.9) -- momentum
        model:updateParameters(0.1) -- learning rate
        model:maxParamNorm(2) -- max norm constraint of weight matrix rows
        model:zeroGradParameters()
    end,
    feedback = dp.Confusion(), -- wrap optim.ConfusionMatrix
    sampler = dp.ShuffleSampler{batch_size = 32},
    progress = true
}
valid = dp.Evaluator{
    feedback = dp.Confusion(), sampler = dp.ShuffleSampler{batch_size = 32}
}
test = dp.Evaluator{
    feedback = dp.Confusion(), sampler = dp.ShuffleSampler{batch_size = 32}
}

xp = dp.Experiment{
    model = cnn,
    optimizer = train, validator = valid, tester = test,
    observer = dp.EarlyStopper{
        error_report = {'validator', 'feedback', 'confusion', 'accuracy'},
        maximize = true, max_epochs = 50
    },
    random_seed = os.time(), max_epochs = 2000
}

local ds = dp.Mnist{input_preprocess = input_preprocess}
xp:run(ds)
