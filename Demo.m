% This is the testing demo of reconstructing HSI image 
% To run the code, you should install Matconvnet first.
% Then, put 'ResultReshape.m','xProblem_Sum.m', and 'xReshape.m' under the 
% path of 'your matlab path\matconvnet\matconvnet-1.0-beta25\matlab\+dagnn\'

clear all
close all

gpu         = 1;

hyperspectralSlices =1:31;

size_input = 64;
stride = size_input/2;
%load reconstruction net
load('.\Model\Harvard_CASSI_Recon_Net.mat')
net = dagnn.DagNN.loadobj(net) ;
net.removeLayer('loss') ;
out = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;
net.mode = 'test';
if gpu
    net.move('gpu');
end  
SZ=[size_input,size_input];
SZ=gpuArray(SZ);
%load mask
load('.\Mask\mask_Cassi_patch64_32stride.mat')
PHI = sparse([  size_input * size_input , size_input * size_input*numel(hyperspectralSlices) ]);
    for ks=1:numel(hyperspectralSlices)
        currentMSlice = M(:,:,ks);
        PHI(1: size_input * size_input, (ks-1)*(size_input * size_input)+1:ks*( size_input * size_input)) = sparse(1: size_input * size_input,1: size_input * size_input,currentMSlice);
    end
%load test image
load('.\test\Harvard36.mat')
[hei,wid,ch] = size(hyper_image); 
moreRow = (floor((542-size_input+stride-1)/stride))*stride + size_input -542;
if(hei>512 || wid>512 )
    CropIndex = [round(hei/2) - 256-30-floor(moreRow/2), round(hei/2) + 255+30+(moreRow-floor(moreRow/2)), ...
        round(wid/2) - 256, round(wid/2) + 255];
else
    CropIndex = [1, hei, ...
        1, wid];
end
hyper_image = hyper_image(CropIndex(1) : CropIndex(2), CropIndex(3) : CropIndex(4) , :);    
% generate HSI path
label_Image = zeros(size_input,size_input,31,1,'single');
count = 1;
kh=1;
kw=1;
[hei,wid,ch] = size(hyper_image);
for x = 1 : stride : hei-size_input+1-numel(hyperspectralSlices)+1
    for y = 1 :stride : wid-size_input+1 
        for ch = 1:numel(hyperspectralSlices)
            label_Image(:,:,ch,count) = hyper_image(x+(ch-1) : x+size_input-1+(ch-1), y : y+size_input-1,ch);
        end
        count=count+1;
        kw=y;
    end
    kh=x;
end
label= hyper_image(1:kh+size_input-1+numel(hyperspectralSlices)-1,1:kw+size_input-1,:);%前30行和后30行均需舍弃（信息采集不完全）
sz = size(label_Image);
sz=[sz,1];
label_Image = gpuArray(label_Image);
label_Image = reshape(label_Image,[sz(1)*sz(2)*sz(3),sz(4)]);
%  All patch have the same mask 
PHI = gpuArray(PHI);
cassiPatch = full(PHI*double(label_Image));
input = im2single(full(transpose(PHI)*sparse(im2double(cassiPatch))));
PHITPHI = transpose(PHI)*PHI;
net.eval({'input', input,'PHITPHI',PHITPHI,'SZ',SZ}) 
output = gather(squeeze(gather(net.vars(out).value)));
    
%reconstruct image
count=1;
result_Image = zeros(size(label));
result_Weight = zeros(size(label));
for x = 1 : stride : hei-size_input+1-numel(hyperspectralSlices)+1
    for y = 1 :stride : wid-size_input+1 
        for ch=1:31
            temp_out = output(:,:,:,count);
            result_Image(x+(ch-1) : x+size_input-1+(ch-1), y : y+size_input-1,ch)=result_Image(x+(ch-1) : x+size_input-1+(ch-1), y : y+size_input-1,ch)...
                +temp_out(:,:,ch);
            result_Weight(x+(ch-1) : x+size_input-1+(ch-1), y : y+size_input-1,ch)=result_Weight(x+(ch-1) : x+size_input-1+(ch-1), y : y+size_input-1,ch)...
                +1;
        end
        count=count+1;
    end
 end
result_Image = result_Image(30+1+floor(moreRow/2):end-30-(moreRow-floor(moreRow/2)),:,:);
result_Weight = result_Weight(30+1+floor(moreRow/2):end-30-(moreRow-floor(moreRow/2)),:,:);
result=result_Image./result_Weight;
label = label(30+1+floor(moreRow/2):end-30-(moreRow-floor(moreRow/2)),:,:);
result=result.*(result>0);
   
   
%PSNR,SSIM,SAM
h = 512;
w = 512;
ground_truth = label;
img_result=result;
psnr=zeros(1,31);
for k = 1:31
    img_max = max(max(ground_truth(:,:,k)));
    err = mean(mean((ground_truth(:,:,k)-img_result(:,:,k)).^2));
    psnr(k) = 10*log10(img_max^2/err);
end
PSNR = mean(psnr);
    
ssim=zeros(1,31);
k1 = 0.01;
k2 = 0.03;
for k = 1:31
    a = 2*mean(mean(img_result(:,:,k))) * mean(mean(ground_truth(:,:,k))) + k1^2;
    x = cov(reshape(img_result(:,:,k), h*w, 1), reshape(ground_truth(:,:,k), h*w, 1));
    b = 2*x(1,2) + k2^2;
    c = mean(mean(img_result(:,:,k)))^2 + mean(mean(ground_truth(:,:,k)))^2 + k1^2;
    d = x(1,1) + x(2,2) + k2^2;
    ssim(k) = a*b/c/d;
end
SSIM = mean(ssim);

tmp = (sum(ground_truth.*img_result, 3) + eps) ...
    ./ (sqrt(sum(ground_truth.^2, 3)) + eps) ./ (sqrt(sum(img_result.^2, 3)) + eps);
SAM = mean2(real(acos(tmp)));    
savedir = ['.\Result'];
if ~exist(savedir,'file')
    mkdir(savedir);
end
save(fullfile(savedir,'result.mat'),'result','label','PSNR','SSIM','SAM');

























