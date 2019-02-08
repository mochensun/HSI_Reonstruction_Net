classdef ResultReshape < dagnn.ElementWise
  %Reshape DagNN ResultReshape layer
  %   the layer for the CASSI System Reconstruction based on patches
  %   Input:column vector
  %   Output: oblique parallelepiped HSI patch
  
  methods
    function outputs = forward(obj, inputs, params)
      sz=inputs{2};
      sz=gather(sz);
	  [sizeX, sizeY, c, bath] = size(inputs{1});
      
      outputs{1} = reshape(inputs{1},[sz(1),sz(2),31,sizeY]);
    end
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz=inputs{2};
	  [sizeX, sizeY, c, bath] = size(derOutputs{1});
      derInputs{1} = reshape(derOutputs{1},[sz(1) * sz(2)*31,bath]);
      derInputs{2}=0;
      derParams=[];
    end
  end
end
