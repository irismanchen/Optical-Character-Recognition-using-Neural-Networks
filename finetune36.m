num_epoch = 20;
classes = 36;
layers = [32*32, 800, classes];
learning_rate = 0.01;

load('../data/nist36_train.mat', 'train_data', 'train_labels')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
load('../data/nist36_valid.mat', 'valid_data', 'valid_labels')

% [W1, b1] = InitializeNetwork(layers);
% load('../data/nist26_model_60iters.mat', 'W', 'b')
% W{2} = W1{2};
% b{2} = b1{2};
% W = reshape(W,[2,1]);
% b = reshape(b,[2,1]);
load('nist36_model.mat', 'W', 'b')
total_train_acc = zeros(1,num_epoch);
total_train_loss = zeros(1,num_epoch);
total_valid_acc = zeros(1,num_epoch);
total_valid_loss = zeros(1,num_epoch);
for j = 1:num_epoch
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);

    [train_acc, train_loss] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    [valid_acc, valid_loss] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    
    total_train_acc(1,j) = train_acc;
    total_train_loss(1,j) = train_loss;
    total_valid_acc(1,j) = valid_acc;
    total_valid_loss(1,j) = valid_loss;
    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc, valid_acc, train_loss, valid_loss)
end
save('nist36_model', 'W', 'b')
save('nist36_accLoss.mat','total_train_acc','total_train_loss','total_valid_acc','total_valid_loss');
plot(total_train_acc);
hold on;
plot(total_valid_acc);
pause;
clf;
plot(total_train_loss);
hold on;
plot(total_valid_loss);