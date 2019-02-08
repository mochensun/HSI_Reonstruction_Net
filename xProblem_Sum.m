classdef xProblem_Sum < dagnn.ElementWise
  %SUM DagNN xProblem_Sum layer
  %   The layer takes the sum of its weighted inputs(xt,x0,z)
  %   Input:(1)xt, the result of previous step
  %         (2)x0, transpose(PHI)*y
  %         (3)z, Regularization result
  %         (4)PHI, transpose(PHI)*PHI
  %   Output:x_t+1

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)%inputs(xt,x0,z,phiTphi),params(deta,eta)
      obj.numInputs = numel(inputs) ;
      deta = params{1};
      eta = params{2};
      if(obj.numInputs == 3)
          x0 = inputs{1};
          z = inputs{2};
          PHITPHI = inputs{3};
          PHITPHI_x0=single(PHITPHI*double(x0));
          outputs{1} = (1-2*deta*eta)*x0-2*deta*PHITPHI_x0+2*deta*x0+2*deta*eta*z;
      elseif(obj.numInputs ==4)
          xt = inputs{1};
          x0 = inputs{2};
          z = inputs{3};
          PHITPHI = inputs{4};
          PHITPHI_xt = single(PHITPHI*double(xt));
          outputs{1} = (1-2*deta*eta)*xt-2*deta*PHITPHI_xt+2*deta*x0+2*deta*eta*z;
      else %
          disp('input errors');
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      tag1=2;%1不更新参数deta和eta， 2更新参数deta和eta
      obj.numInputs = numel(inputs) ;
      deta = params{1};
      eta = params{2};
      if(obj.numInputs == 3)%((x0,z,phiTphi))
          x0 = inputs{1};
          z = inputs{2};
          PHITPHI=inputs{3};
          PHITPHI_derOut1=single(PHITPHI*double(derOutputs{1}));          
          derInputs{1} = (1 - 2*deta*eta+2*deta)*derOutputs{1}-2*deta*PHITPHI_derOut1;%x0梯度
          derInputs{2} = 2*deta*eta*derOutputs{1};%z梯度      
          derInputs{3} = 0;
          if(tag1==1)
              derParams{2} =0;
              derParams{1} =0;
          else
              PHITPHI_x0=single(PHITPHI*double(x0));
              darg1=2*((1-eta)*x0-PHITPHI_x0+eta*z).*derOutputs{1};
              sz=[size(deta),1,1,1];
              sz = sz(1:4) ;
              for k = find(sz == 1)
                darg1 = sum(darg1, k) ;
              end
              derParams{1} = darg1;%deta梯度
              darg2=2*(-deta*x0+deta*z).*derOutputs{1};
              sz=[size(eta),1,1,1];
              sz = sz(1:4) ;
              for k = find(sz == 1)
                darg2 = sum(darg2, k) ;
              end
              derParams{2} = darg2;%eta梯度  
          end
      elseif(obj.numInputs ==4)%(xt,x0,z,phiTphi)
          xt = inputs{1};
          x0 = inputs{2};
          z = inputs{3};
          PHITPHI=inputs{4};
          PHITPHI_derOut1=single(PHITPHI*double(derOutputs{1}));     
          derInputs{1} = (1- 2*deta*eta)*derOutputs{1}-2*deta*PHITPHI_derOut1;%xt梯度
          derInputs{2} = 2*deta*derOutputs{1};%x0梯度
          derInputs{3} = 2*deta*eta*derOutputs{1};%z梯度
          derInputs{4} = 0;
          if(tag1==1)
              derParams{2} =0;
              derParams{1} =0;
          else
              PHITPHI_xt=single(PHITPHI*double(xt));
              darg1=2*(-eta*xt-PHITPHI_xt+x0+eta*z).*derOutputs{1};
              sz=[size(deta),1,1,1];
              sz = sz(1:4) ;
              for k = find(sz == 1)
                darg1 = sum(darg1, k) ;
              end
              derParams{1} = darg1;%deta梯度
              darg2=2*(-deta*xt+deta*z).*derOutputs{1};
              sz=[size(eta),1,1,1];
              sz = sz(1:4) ;
              for k = find(sz == 1)
                darg2 = sum(darg2, k) ;
              end
              derParams{2} = darg2;%eta梯度 
          end
      else %
          disp('input errors');
      end
      
    end
    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = xProblem_Sum(varargin)
      obj.load(varargin) ;
    end
  end
end
