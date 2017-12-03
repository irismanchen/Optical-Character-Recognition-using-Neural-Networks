function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

% Your code here
layerNum = length(layers);
W = cell([layerNum-1, 1]);
b = cell([layerNum-1, 1]);
for i=1:layerNum-1
    W{i} = normrnd(0,0.01,[layers(i+1),layers(i)]);
    b{i} = ones([layers(i+1),1]);
    %n = layers(i);
    %W{i} = ones([layers(i+1),layers(i)])/n;
    %b{i} = ones([layers(i+1),1]);
C = size(b{end},1);
assert(size(W{1},2) == 1024, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'W{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');

end
