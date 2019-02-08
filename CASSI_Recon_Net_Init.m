function net =CASSI_Recon_Net_Init()
% Create DAGNN object
net = dagnn.DagNN();

blockNum1 = 1;%the first branch
blockNum2 = 1;%the second branch
inVar1 = 'input';%x0=AT*y
channel= 31; % Hyperspectrum image
tag1=1;%the first branch
tag2=2;%the second branch

PHITPHI='PHITPHI';
SZ='SZ';
x0 = inVar1;
inVar2=inVar1;
[net, inVar2, blockNum2] = addxReshape(net, blockNum2, {inVar2,SZ},tag2);
%第二个分支
x=inVar2;
dims   = [3,3,channel,64];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
[net, inVar2, blockNum2] = addReLU(net, blockNum2, inVar2,tag2); 
dims   = [3,3,64,channel];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
[net, inVar2, blockNum2] = addSum(net, blockNum2, {inVar2,x},tag2);
dims   = [1,1,channel,channel];
pad    = [0,0];
stride = [1,1];
lr     = [1,1];
[net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
[net, inVar2, blockNum2] = addxReshape(net, blockNum2, {inVar2,SZ},tag2);
z=inVar2;
%第一个分支，求x1
[net, inVar1, blockNum1] = addxProblem_Sum(net, blockNum1, {x0,z,PHITPHI},tag1);

for i=2:9
    %第二个分支
    inVar2=inVar1;
    [net, inVar2, blockNum2] = addxReshape(net, blockNum2, {inVar2,SZ},tag2);%立方体
    x=inVar2;
    dims   = [3,3,channel,64];
    pad    = [1,1];
    stride = [1,1];
    lr     = [1,1];
    [net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
    [net, inVar2, blockNum2] = addReLU(net, blockNum2, inVar2,tag2); 
    dims   = [3,3,64,channel];
    pad    = [1,1];
    stride = [1,1];
    lr     = [1,1];
    [net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
    [net, inVar2, blockNum2] = addSum(net, blockNum2, {inVar2,x},tag2);   
    dims   = [1,1,channel,channel];
    pad    = [0,0];
    stride = [1,1];
    lr     = [1,1];
    [net, inVar2, blockNum2] = addConv(net, blockNum2, inVar2, dims, pad, stride, lr,tag2);
    
    [net, inVar2, blockNum2] = addxReshape(net, blockNum2, {inVar2,SZ},tag2);
    z=inVar2;
    
    %第一个分支xt
    [net, inVar1, blockNum1] = addxProblem_Sum(net, blockNum1, {inVar1,x0,z,PHITPHI},tag1);
    
    
end
%第一个分支
[net, inVar1, blockNum1] = addResultReshape(net, blockNum1, {inVar1,SZ},tag1);%立方体   
%add loss
outputName = 'prediction';
net.renameVar(inVar1,outputName)
% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{}); 
net.vars(net.getVarIndex('prediction')).precious = 1;

end

function [net, inVar, blockNum] = addResultReshape(net, blockNum, inVar,tag)

outVar   = sprintf(['resultreshape' num2str(tag)  '_%d'], blockNum);
layerCur = sprintf(['resultreshape' num2str(tag)  '_%d'], blockNum);

block = dagnn.ResultReshape();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


function [net, inVar, blockNum] = addxReshape(net, blockNum, inVar,tag)

outVar   = sprintf(['xreshape' num2str(tag)  '_%d'], blockNum);
layerCur = sprintf(['xreshape' num2str(tag)  '_%d'], blockNum);

block = dagnn.xReshape();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a sum layer
function [net, inVar, blockNum] = addxProblem_Sum(net, blockNum, inVar,tag)
trainMethod = 'adam';
outVar   = sprintf(['xProblem_Sum' num2str(tag),'_%d'], blockNum);
layerCur = sprintf(['xProblem_Sum' num2str(tag),'_%d'], blockNum);

block    = dagnn.xProblem_Sum();

params={['deta_',num2str(blockNum)],  ['eta_',num2str(blockNum),]};
net.addLayer(layerCur, block, inVar, {outVar},params);
f  = net.getParamIndex(params{1}) ;
net.params(f).value        =0.1;
net.params(f).learningRate = 1;
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f  = net.getParamIndex(params{2}) ;
net.params(f).value        = 1;
net.params(f).learningRate = 1;
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar,tag)

outVar   = sprintf(['sum' num2str(tag) '_%d'], blockNum);
layerCur = sprintf(['sum' num2str(tag) '_%d'], blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar,tag)

outVar   = sprintf([ 'relu' num2str(tag) '_%d'], blockNum);
layerCur = sprintf(['relu' num2str(tag)  '_%d'], blockNum);

block    = dagnn.ReLU('leak',0);

net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr,tag)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf(['conv' num2str(tag) '_%d'], blockNum);
layerCur    = sprintf(['conv' num2str(tag) '_%d'], blockNum);


convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
        'hasBias', true, 'opts', convOpts);
params={[layerCur '_f'], [layerCur '_b']};


net.addLayer(layerCur, convBlock, {inVar}, {outVar},params);

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xvaier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end



