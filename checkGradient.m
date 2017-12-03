load('../data/nist26_train.mat', 'train_data', 'train_labels')
classes = 26;
layers = [32*32, 400, classes];
[W, b] = InitializeNetwork(layers);
epsilon = 1e-4;
thresh = 1e-3;
num_layers = length(layers) - 1;
check_num = 1000;
sum_error = 0;
for k = 1:check_num
    rnum = unidrnd(size(train_data,1));
    [~, act_h, act_a] = Forward(W, b, train_data(rnum, :)');
    [grad_W, grad_b] = Backward(W, b, train_data(rnum,:)', train_labels(rnum,:)', act_h, act_a);
    [~, loss] = ComputeAccuracyAndLoss(W, b, train_data(rnum,:), train_labels(rnum,:));
    for i = 1:num_layers
        [h,w] = size(W{i});
        y = unidrnd(h);
        x = unidrnd(w);
        % For b
        [b_plus,b_minus] = deal(b);
        b_plus{i}(y) = b_plus{i}(y) + epsilon;
        [~, loss_plus2] = ComputeAccuracyAndLoss(W, b_plus, train_data(rnum,:), train_labels(rnum,:));
        b_minus{i}(y) = b_minus{i}(y) - epsilon;
        [~, loss_minus2] = ComputeAccuracyAndLoss(W, b_minus, train_data(rnum,:), train_labels(rnum,:));
        grad2 = (loss_plus2-loss_minus2)/(2*epsilon);
        assert(abs(grad_b{i}(y)-grad2)<thresh,'b grad is wrong');
        sum_error = sum_error + abs(grad2-grad_b{i}(y));
        % For W
        [W_plus,W_minus] = deal(W);
        W_plus{i}(y,x) = W_plus{i}(y,x) + epsilon;
        [~, loss_plus1] = ComputeAccuracyAndLoss(W_plus, b, train_data(rnum,:), train_labels(rnum,:));
        W_minus{i}(y,x) = W_minus{i}(y,x) - epsilon;
        [~, loss_minus1] = ComputeAccuracyAndLoss(W_minus, b, train_data(rnum,:), train_labels(rnum,:));
        grad1 = (loss_plus1-loss_minus1)/(2*epsilon);
        assert(abs(grad_W{i}(y,x)-grad1)<thresh,'W grad is wrong');
        sum_error = sum_error + abs(grad1-grad_W{i}(y,x));
    end
end
error = abs(sum_error) / (check_num*num_layers*2);
assert(error < thresh);