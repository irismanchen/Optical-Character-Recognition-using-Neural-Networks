num_epoch = 200;
classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.001;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

[W, b] = InitializeNetwork(layers);
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
save('nist26_model_epoch200', 'W', 'b')
save('nist26_accloss_epoch200.mat','total_train_acc','total_train_loss','total_valid_acc','total_valid_loss');
plot(total_train_acc);
hold on;
plot(total_valid_acc);
pause;
clf;
plot(total_train_loss);
hold on;
plot(total_valid_loss); 
