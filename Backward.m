function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer post and pre activations when forward
%     forward propogating the input smaple 'X'
N = size(X,1);
H = size(W{1},1);
C = size(b{end},1);
assert(size(W{1},2) == N, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'W{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');
assert(all(size(act_a{1}) == [H,1]), 'act_a{1} must be of size [H,1]');
assert(all(size(act_h{end}) == [C,1]), 'act_h{end} must be of size [C,1]');

% Your code here
layerNum = size(W,1);
grad_W = cell([layerNum, 1]);
grad_b = cell([layerNum, 1]);
outputLayerDelta = act_h{end}-Y;
grad_W{end} = outputLayerDelta*act_h{end-1}';
grad_b{end} = outputLayerDelta;
next_delta = outputLayerDelta;
for i=layerNum-1:-1:1
    cur_deriv = act_h{i}.*(1-act_h{i});
    cur_delta = cur_deriv.*(W{i+1}'*next_delta);
    grad_b{i} = cur_delta;
    if i>1
        grad_W{i} = cur_delta*act_h{i-1}';
    else
        grad_W{i} = cur_delta*X';
    end
    next_delta = cur_delta;
end

assert(size(grad_W{1},2) == N, 'grad_W{1} must be of size [H,N]');
assert(size(grad_W{end},1) == C, 'grad_W{end} must be of size [C,N]');
assert(size(grad_b{1},1) == H, 'grad_b{1} must be of size [H,1]');
assert(size(grad_b{end},1) == C, 'grad_b{end} must be of size [C,1]');

end
