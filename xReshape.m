classdef xReshape < dagnn.ElementWise
  %Reshape DagNN xReshape layer
  %   the layer for the CASSI System Reconstruction based on patches
  %   (1)Input: column vector; Output: pixel-Aligned HSI patch
  %   (1)Input: pixel-Aligned HSI patch; Output: column vector
  
  methods
    function outputs = forward(obj, inputs, params)
      sz=inputs{2};
      sz=gather(sz);
	  [sizeX,sizeY,c,bath] = size(inputs{1});
      if (sizeX~=(sz(1)+30))
		temp = reshape(inputs{1},[sz(1),sz(2),31,sizeY]);
        temp(sz(1)+1:sz(1)+30,:,:,:)=0;
        for ch=1:31
            temp(ch:sz(1)+ch-1,:,ch,:)=temp(1:sz(1),:,ch,:);
            temp(1:ch-1,:,ch,:)=0;
        end
        outputs{1}=temp;
      else
        temp = inputs{1};   
        for ch=1:31
            temp(1:sz(1),:,ch,:)=temp(ch:ch+sz(1)-1,:,ch,:);
        end
        temp(end-30+1:end,:,:,:)=[];
		outputs{1} = reshape(temp,[sz(1) * sz(2)*31,bath]);
	  end 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz=inputs{2};
      sz=gather(sz);
	  [sizeX,sizeY,c,bath] = size(derOutputs{1});
      if (sizeX~=(sz(1)+30))
		temp = reshape(derOutputs{1},[sz(1),sz(2),31,sizeY]);
        temp(sz(1)+1:sz(1)+30,:,:,:)=0;
        for ch=1:31
            temp(ch:sz(1)+ch-1,:,ch,:)=temp(1:sz(1),:,ch,:);
            temp(1:ch-1,:,ch,:)=0;
        end
        derInputs{1}=temp;
      else
        temp = derOutputs{1};   
        for ch=1:31
            temp(1:sz(1),:,ch,:)=temp(ch:ch+sz(1)-1,:,ch,:);
        end
        temp(end-30+1:end,:,:,:)=[];
		derInputs{1} = reshape(temp,[sz(1) * sz(2)*31,bath]);
	  end 
      derInputs{2}=0;
      derParams=[];
    end
  end
end
